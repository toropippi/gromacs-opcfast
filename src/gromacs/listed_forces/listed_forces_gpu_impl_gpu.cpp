/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements helper functions for GPU listed forces (bonded)
 *
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include <array>
#include <vector>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/ifunc.h"

#include "bonded.h"
#include "listed_forces_gpu_impl.h"

namespace gmx
{
// ---- ListedForcesGpu::Impl

static int chooseSubGroupSizeForDevice(const DeviceInformation& deviceInfo)
{
    if (deviceInfo.supportedSubGroupSizes.size() == 1)
    {
        return deviceInfo.supportedSubGroupSizes[0];
    }
    else if (deviceInfo.supportedSubGroupSizes.size() > 1)
    {
        switch (deviceInfo.deviceVendor)
        {
            case DeviceVendor::Intel: return 32;
            case DeviceVendor::PoclCpu: return 32;
            default:
                GMX_RELEASE_ASSERT(false, "Flexible sub-groups only supported for Intel GPUs");
                return 0;
        }
    }
    else
    {
        GMX_RELEASE_ASSERT(false, "Device has no known supported sub-group sizes");
        return 0;
    }
}

#if GMX_GPU_CUDA
/*! \brief Precompute the 16 bicubic coefficients for every cell of every CMAP grid.
 *
 * This follows cmap_dihs() exactly, including degree-scaled first and mixed derivatives.
 */
std::vector<float> makeCmapGpuCoefficients(const gmx_cmap_t& cmapGrid)
{
    if (cmapGrid.cmapdata.empty())
    {
        return {};
    }

    const int gridSpacing = cmapGrid.grid_spacing;
    GMX_RELEASE_ASSERT(gridSpacing > 1, "A CUDA CMAP grid needs at least two points per dimension");

    constexpr int c_numCoefficients = 16;
    const int     valuesPerMap      = gridSpacing * gridSpacing * c_numCoefficients;
    std::vector<float> coefficients(cmapGrid.cmapdata.size() * valuesPerMap);
    const int*         coefficientMatrix = cmapCoefficientMatrix();
    const float        dx                = 360.0F / gridSpacing;

    for (std::size_t map = 0; map < cmapGrid.cmapdata.size(); map++)
    {
        const auto& values = cmapGrid.cmapdata[map].cmap;
        GMX_RELEASE_ASSERT(values.size() == static_cast<std::size_t>(4 * gridSpacing * gridSpacing),
                           "Inconsistent CUDA CMAP grid size");

        for (int iphi1 = 0; iphi1 < gridSpacing; iphi1++)
        {
            const int ip1p1 = (iphi1 + 1) % gridSpacing;
            for (int iphi2 = 0; iphi2 < gridSpacing; iphi2++)
            {
                const int ip2p1 = (iphi2 + 1) % gridSpacing;
                const std::array<int, 4> positions = {
                    iphi1 * gridSpacing + iphi2,
                    ip1p1 * gridSpacing + iphi2,
                    ip1p1 * gridSpacing + ip2p1,
                    iphi1 * gridSpacing + ip2p1
                };

                std::array<float, c_numCoefficients> tx = {};
                for (int corner = 0; corner < 4; corner++)
                {
                    const int valueIndex = 4 * positions[corner];
                    tx[corner]           = values[valueIndex];
                    tx[corner + 4]       = values[valueIndex + 1] * dx;
                    tx[corner + 8]       = values[valueIndex + 2] * dx;
                    tx[corner + 12]      = values[valueIndex + 3] * dx * dx;
                }

                const std::size_t cellOffset = map * valuesPerMap
                                               + (iphi1 * gridSpacing + iphi2) * c_numCoefficients;
                for (int coefficient = 0; coefficient < c_numCoefficients; coefficient++)
                {
                    float value = 0.0F;
                    for (int input = 0; input < c_numCoefficients; input++)
                    {
                        value += coefficientMatrix[input * c_numCoefficients + coefficient] * tx[input];
                    }
                    coefficients[cellOffset + coefficient] = value;
                }
            }
        }
    }
    return coefficients;
}
#endif

ListedForcesGpu::Impl::Impl(const gmx_ffparams_t& ffparams,
                            const float           coulombConstant,
                            const float           fudgeQQ,
                            const int             numEnergyGroupsForListedForces,
                            const DeviceContext&  deviceContext,
                            const DeviceStream&   deviceStream,
                            gmx_wallcycle*        wcycle) :
    deviceContext_(deviceContext), deviceStream_(deviceStream)
{
    GMX_RELEASE_ASSERT(numEnergyGroupsForListedForces == 1,
                       "Only a single energy group is supported with listed forces on GPU");

    GMX_RELEASE_ASSERT(deviceStream.isValid(),
                       "Can't run GPU version of bonded forces in stream that is not valid.");

    static_assert(
            c_threadsBondedPerBlock >= c_numShiftVectors,
            "Threads per block in GPU bonded must be >= c_numShiftVectors for the virial kernel "
            "(calcVir=true)");

    wcycle_ = wcycle;

    allocateDeviceBuffer(&d_forceParams_, ffparams.numTypes(), deviceContext_);
    // This could be an async transfer (if the source is pinned), so
    // long as it uses the same stream as the kernels and we are happy
    // to consume additional pinned pages.
    copyToDeviceBuffer(&d_forceParams_,
                       ffparams.iparams.data(),
                       0,
                       ffparams.numTypes(),
                       deviceStream_,
                       GpuApiCallBehavior::Sync,
                       nullptr);
#if GMX_GPU_CUDA
    const std::vector<float> cmapCoefficients = makeCmapGpuCoefficients(ffparams.cmap_grid);
    if (!cmapCoefficients.empty())
    {
        allocateDeviceBuffer(&d_cmapCoefficients_, cmapCoefficients.size(), deviceContext_);
        copyToDeviceBuffer(&d_cmapCoefficients_,
                           cmapCoefficients.data(),
                           0,
                           cmapCoefficients.size(),
                           deviceStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
        kernelParams_.cmapGridSpacing = ffparams.cmap_grid.grid_spacing;
        kernelParams_.cmapCoefficientsPerMap =
                16 * ffparams.cmap_grid.grid_spacing * ffparams.cmap_grid.grid_spacing;
        kernelBuffers_.d_cmapCoefficients = d_cmapCoefficients_;
    }
#endif
    vTot_.resize(static_cast<int>(InteractionFunction::Count));
    allocateDeviceBuffer(&d_vTot_, static_cast<int>(InteractionFunction::Count), deviceContext_);
    clearDeviceBufferAsync(&d_vTot_, 0, static_cast<int>(InteractionFunction::Count), deviceStream_);

    kernelParams_.coulombConstant               = coulombConstant;
    kernelParams_.fudgeQQ                       = fudgeQQ;
    kernelParams_.electrostaticsScaleFactor = coulombConstant * fudgeQQ;
    kernelBuffers_.d_forceParams            = d_forceParams_;
    kernelBuffers_.d_vTot                   = d_vTot_;
    for (int i = 0; i < numFTypesOnGpu; i++)
    {
        kernelBuffers_.d_iatoms[i]       = nullptr;
        kernelParams_.fTypeRangeStart[i] = 0;
        kernelParams_.fTypeRangeEnd[i]   = -1;
    }

    const DeviceInformation& deviceInfo = deviceContext.deviceInfo();
    deviceSubGroupSize_                 = chooseSubGroupSizeForDevice(deviceInfo);
    GMX_RELEASE_ASSERT(std::find(deviceInfo.supportedSubGroupSizes.begin(),
                                 deviceInfo.supportedSubGroupSizes.end(),
                                 deviceSubGroupSize_)
                               != deviceInfo.supportedSubGroupSizes.end(),
                       "Device does not support selected sub-group size");

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    kernelLaunchConfig_.blockSize[0] = c_threadsBondedPerBlock;
    kernelLaunchConfig_.blockSize[1] = 1;
    kernelLaunchConfig_.blockSize[2] = 1;
    kernelLaunchConfig_.gridSize[0] = (fTypeRangeEnd + c_threadsBondedPerBlock) / c_threadsBondedPerBlock;
    kernelLaunchConfig_.gridSize[1]      = 1;
    kernelLaunchConfig_.gridSize[2]      = 1;
    kernelLaunchConfig_.sharedMemorySize = c_numShiftVectors * sizeof(Float3);
}

ListedForcesGpu::Impl::~Impl()
{
    try
    {
        deviceStream_.synchronize();
        for (InteractionFunction fType : fTypesOnGpu)
        {
            const int ft = static_cast<int>(fType);
            if (d_iAtoms_[ft])
            {
                freeDeviceBuffer(&d_iAtoms_[ft]);
                d_iAtoms_[ft] = nullptr;
            }
        }

        freeDeviceBuffer(&d_forceParams_);
#if GMX_GPU_CUDA
        if (d_cmapCoefficients_)
        {
            freeDeviceBuffer(&d_cmapCoefficients_);
        }
#endif
        freeDeviceBuffer(&d_vTot_);
    }
    catch (gmx::InternalError& e)
    {
        fprintf(stderr, "Internal error in destructor of ListedForcesGpu: %s\n", e.what());
    }
}

//! Return whether function type \p fType in \p idef has perturbed interactions
static bool fTypeHasPerturbedEntries(const InteractionDefinitions& idef, int fType)
{
    GMX_ASSERT(idef.ilsort == ilsortNO_FE || idef.ilsort == ilsortFE_SORTED,
               "Perturbed interactions should be sorted here");

    const InteractionList& ilist = idef.il[fType];

    return (idef.ilsort != ilsortNO_FE && idef.numNonperturbedInteractions[fType] != ilist.size());
}

//! Converts \p src with atom indices in state order to \p dest in nbnxn order
static void convertIlistToNbnxnOrder(const InteractionList& src,
                                     HostInteractionList*   dest,
                                     int                    numAtomsPerInteraction,
                                     ArrayRef<const int>    nbnxnAtomOrder)
{
    GMX_ASSERT(src.empty() || !nbnxnAtomOrder.empty(), "We need the nbnxn atom order");

    dest->iatoms.resize(src.size());

#pragma omp parallel for num_threads(gmx_omp_nthreads_get(ModuleMultiThread::Bonded)) schedule(static)
    for (int i = 0; i < src.size(); i += 1 + numAtomsPerInteraction)
    {
        dest->iatoms[i] = src.iatoms[i];
        for (int a = 0; a < numAtomsPerInteraction; a++)
        {
            dest->iatoms[i + 1 + a] = nbnxnAtomOrder[src.iatoms[i + 1 + a]];
        }
    }
}

//! Returns \p input rounded up to the closest multiple of \p factor.
static inline int roundUpToFactor(const int input, const int factor)
{
    GMX_ASSERT(factor > 0, "The factor to round up to must be > 0.");

    int remainder = input % factor;

    if (remainder == 0)
    {
        return (input);
    }
    return (input + (factor - remainder));
}

void ListedForcesGpu::Impl::updateHaveInteractions(const InteractionDefinitions& idef)
{
    haveInteractions_ = false;

    for (InteractionFunction fType : fTypesOnGpu)
    {
        /* Perturbation is not implemented in the GPU bonded kernels.
         * But instead of doing all interactions on the CPU, we can
         * still easily handle the types that have no perturbed
         * interactions on the GPU. */
        const int ft = static_cast<int>(fType);
        if (!idef.il[ft].empty() && !fTypeHasPerturbedEntries(idef, ft))
        {
            haveInteractions_ = true;
            return;
        }
    }
}


// TODO Consider whether this function should be a factory method that
// makes an object that is the only one capable of the device
// operations needed for the lifetime of an interaction list. This
// would be harder to misuse than ListedForcesGpu, and exchange the problem
// of naming this method for the problem of what to name the
// BondedDeviceInteractionListHandler type.

/*! Divides bonded interactions over threads and GPU.
 *  The bonded interactions are assigned by interaction type to GPU threads. The interaction
 *  types are assigned in blocks sized as kernelParams_.deviceSubGroupSize. The beginning and end
 *  (thread index) of each interaction type are stored in kernelParams_. Pointers to the relevant
 *  data structures on the GPU are also stored in kernelParams_.
 */
void ListedForcesGpu::Impl::updateInteractionListsAndDeviceBuffers(ArrayRef<const int> nbnxnAtomOrder,
                                                                   const InteractionDefinitions& idef,
                                                                   DeviceBuffer<Float4> d_xqPtr,
                                                                   DeviceBuffer<RVec>   d_fPtr,
                                                                   DeviceBuffer<RVec>   d_fShiftPtr)
{
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::GpuBondedListUpdate);
    bool haveGpuInteractions = false;
    int  fTypesCounter       = 0;

    for (InteractionFunction fType : fTypesOnGpu)
    {
        const int ft    = static_cast<int>(fType);
        auto&     iList = iLists_[ft];

        /* Perturbation is not implemented in the GPU bonded kernels.
         * But instead of doing all interactions on the CPU, we can
         * still easily handle the types that have no perturbed
         * interactions on the GPU. */
        if (!idef.il[ft].empty() && !fTypeHasPerturbedEntries(idef, ft))
        {
            haveGpuInteractions = true;

            convertIlistToNbnxnOrder(idef.il[ft], &iList, NRAL(fType), nbnxnAtomOrder);
        }
        else
        {
            iList.iatoms.clear();
        }

        // Update the device if necessary. This can leave some
        // allocations on the device when the host size decreases to
        // zero, which is OK, since we deallocate everything at the
        // end.
        if (iList.size() > 0)
        {
            int newListSize;
            reallocateDeviceBuffer(
                    &d_iAtoms_[ft], iList.size(), &newListSize, &d_iAtomsAlloc_[ft], deviceContext_);

            copyToDeviceBuffer(&d_iAtoms_[ft],
                               iList.iatoms.data(),
                               0,
                               iList.size(),
                               deviceStream_,
                               GpuApiCallBehavior::Async,
                               nullptr);
        }
        kernelParams_.fTypesOnGpu[fTypesCounter] = static_cast<InteractionFunction>(ft);
        int numBonds = iList.size() / (interaction_function[ft].nratoms + 1);
        kernelParams_.numFTypeBonds[fTypesCounter] = numBonds;
        kernelBuffers_.d_iatoms[fTypesCounter]     = d_iAtoms_[ft];
        if (fTypesCounter == 0)
        {
            kernelParams_.fTypeRangeStart[fTypesCounter] = 0;
        }
        else
        {
            kernelParams_.fTypeRangeStart[fTypesCounter] =
                    kernelParams_.fTypeRangeEnd[fTypesCounter - 1] + 1;
        }
        kernelParams_.fTypeRangeEnd[fTypesCounter] = kernelParams_.fTypeRangeStart[fTypesCounter]
                                                     + roundUpToFactor(numBonds, deviceSubGroupSize_) - 1;

        GMX_ASSERT(numBonds > 0
                           || kernelParams_.fTypeRangeEnd[fTypesCounter]
                                      <= kernelParams_.fTypeRangeStart[fTypesCounter],
                   "Invalid GPU listed forces setup. numBonds must be > 0 if there are threads "
                   "allocated to do work on that interaction function type.");
        GMX_ASSERT(
                kernelParams_.fTypeRangeStart[fTypesCounter] % deviceSubGroupSize_ == 0
                        && (kernelParams_.fTypeRangeEnd[fTypesCounter] + 1) % deviceSubGroupSize_ == 0,
                "The bonded interactions must be assigned to the GPU in blocks of sub-group size.");

        fTypesCounter++;
    }

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];
    kernelLaunchConfig_.gridSize[0] = (fTypeRangeEnd + c_threadsBondedPerBlock) / c_threadsBondedPerBlock;

    d_xq_     = d_xqPtr;
    d_f_      = d_fPtr;
    d_fShift_ = d_fShiftPtr;

    kernelBuffers_.d_forceParams = d_forceParams_;
    kernelBuffers_.d_vTot        = d_vTot_;

    GMX_RELEASE_ASSERT(haveGpuInteractions == haveInteractions_,
                       "inconsistent haveInteractions flags encountered.");

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::GpuBondedListUpdate);
}

void ListedForcesGpu::Impl::setPbc(PbcType pbcType, const matrix box, bool canMoleculeSpanPbc)
{
    PbcAiuc pbcAiuc;
    setPbcAiuc(canMoleculeSpanPbc ? numPbcDimensions(pbcType) : 0, box, &pbcAiuc);
    kernelParams_.pbcAiuc = pbcAiuc;
}

bool ListedForcesGpu::Impl::haveInteractions() const
{
    return haveInteractions_;
}

void ListedForcesGpu::Impl::launchEnergyTransfer()
{
    GMX_ASSERT(haveInteractions_,
               "No GPU bonded interactions, so no energies will be computed, so transfer should "
               "not be called");
    wallcycle_sub_start_nocount(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    // TODO add conditional on whether there has been any compute (and make sure host buffer doesn't contain garbage)
    float* h_vTot = vTot_.data();
    copyFromDeviceBuffer(h_vTot,
                         &d_vTot_,
                         0,
                         static_cast<int>(InteractionFunction::Count),
                         deviceStream_,
                         GpuApiCallBehavior::Async,
                         nullptr);
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
}

void ListedForcesGpu::Impl::waitAccumulateEnergyTerms(gmx_enerdata_t* enerd)
{
    GMX_ASSERT(haveInteractions_,
               "No GPU bonded interactions, so no energies will be computed or transferred, so "
               "accumulation should not occur");

    wallcycle_start(wcycle_, WallCycleCounter::WaitGpuBonded);
    deviceStream_.synchronize();
    wallcycle_stop(wcycle_, WallCycleCounter::WaitGpuBonded);

    for (InteractionFunction fType : fTypesOnGpu)
    {
        const int ft = static_cast<int>(fType);
        if (fType != InteractionFunction::LennardJones14
            && fType != InteractionFunction::LennardJonesCoulomb14Q
            && fType != InteractionFunction::Coulomb14)
        {
            enerd->term[ft] += vTot_[ft];
        }
    }

    // Note: We do not support energy groups here
    gmx_grppairener_t* grppener = &enerd->grpp;
    grppener->energyGroupPairTerms[NonBondedEnergyTerms::LJ14][0] +=
            vTot_[static_cast<int>(InteractionFunction::LennardJones14)]
            + vTot_[static_cast<int>(InteractionFunction::LennardJonesCoulomb14Q)];
    grppener->energyGroupPairTerms[NonBondedEnergyTerms::Coulomb14][0] +=
            vTot_[static_cast<int>(InteractionFunction::Coulomb14)];
}

void ListedForcesGpu::Impl::clearEnergies()
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start_nocount(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    clearDeviceBufferAsync(&d_vTot_, 0, static_cast<int>(InteractionFunction::Count), deviceStream_);
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

// ---- ListedForcesGpu

ListedForcesGpu::ListedForcesGpu(const gmx_ffparams_t& ffparams,
                                 const float           coulombConstant,
                                 const float           fudgeQQ,
                                 const int             numEnergyGroupsForListedForces,
                                 const DeviceContext&  deviceContext,
                                 const DeviceStream&   deviceStream,
                                 gmx_wallcycle*        wcycle) :
    impl_(new Impl(ffparams, coulombConstant, fudgeQQ, numEnergyGroupsForListedForces, deviceContext, deviceStream, wcycle))
{
}

ListedForcesGpu::~ListedForcesGpu() = default;

void ListedForcesGpu::updateHaveInteractions(const InteractionDefinitions& idef)
{
    impl_->updateHaveInteractions(idef);
}

void ListedForcesGpu::updateInteractionListsAndDeviceBuffers(ArrayRef<const int> nbnxnAtomOrder,
                                                             const InteractionDefinitions& idef,
                                                             NBAtomDataGpu* nbnxmAtomDataGpu)
{
    impl_->updateInteractionListsAndDeviceBuffers(
            nbnxnAtomOrder, idef, nbnxmAtomDataGpu->xq, nbnxmAtomDataGpu->f, nbnxmAtomDataGpu->fShift);
}

void ListedForcesGpu::setPbc(PbcType pbcType, const matrix box, bool canMoleculeSpanPbc)
{
    impl_->setPbc(pbcType, box, canMoleculeSpanPbc);
}

bool ListedForcesGpu::haveInteractions() const
{
    return impl_->haveInteractions();
}

void ListedForcesGpu::setPbcAndlaunchKernel(PbcType                  pbcType,
                                            const matrix             box,
                                            bool                     canMoleculeSpanPbc,
                                            const gmx::StepWorkload& stepWork)
{
    setPbc(pbcType, box, canMoleculeSpanPbc);
    launchKernel(stepWork);
}

void ListedForcesGpu::launchEnergyTransfer()
{
    impl_->launchEnergyTransfer();
}

void ListedForcesGpu::waitAccumulateEnergyTerms(gmx_enerdata_t* enerd)
{
    impl_->waitAccumulateEnergyTerms(enerd);
}

void ListedForcesGpu::clearEnergies()
{
    impl_->clearEnergies();
}

} // namespace gmx
