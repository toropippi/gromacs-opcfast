/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#include "gmxpre.h"

#include "config.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdlib/gpuforcereduction_impl_internal.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/opc_vsite_gpu.h"
#include "gromacs/mdlib/opc_vsite_support.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc_aiuc.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/test_hardware_environment.h"

namespace gmx::test
{
namespace
{

constexpr float c_tolerance = 2.0e-6F;

std::vector<OpcVsiteGpuData> threeOpcSites()
{
    std::vector<OpcVsiteGpuData> sites;
    for (int molecule = 0; molecule < 3; ++molecule)
    {
        const int offset = 4 * molecule;
        sites.push_back({ offset, offset + 1, offset + 2, offset + 3, 0.1477224F, 0.1477224F });
    }
    return sites;
}

std::vector<RVec> testCoordinates()
{
    return { { 0.10F, 0.20F, 0.30F },
             { 0.19F, 0.20F, 0.30F },
             { 0.07F, 0.28F, 0.30F },
             { 0.10F, 0.20F, 0.30F },
             { 3.10F, 4.20F, 5.30F },
             { 3.19F, 4.20F, 5.30F },
             { 3.07F, 4.28F, 5.30F },
             { 3.10F, 4.20F, 5.30F },
             { 8.90F, 1.20F, 2.30F },
             { 8.99F, 1.20F, 2.30F },
             { 8.87F, 1.28F, 2.30F },
             { 8.90F, 1.20F, 2.30F } };
}

InteractionLists makeCpuVsiteLists(const std::vector<OpcVsiteGpuData>& sites)
{
    InteractionLists lists;
    for (const auto& site : sites)
    {
        auto& iatoms = lists[InteractionFunction::VirtualSite3].iatoms;
        iatoms.insert(iatoms.end(), { 0, site.virtualSite, site.oxygen, site.hydrogen1, site.hydrogen2 });
    }
    return lists;
}

void expectVectorsNear(const std::vector<RVec>& actual, const std::vector<RVec>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t atom = 0; atom < actual.size(); ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(actual[atom][d], expected[atom][d], c_tolerance)
                    << "atom=" << atom << " dimension=" << d;
        }
    }
}

std::array<double, DIM> totalForce(const std::vector<RVec>& forces)
{
    std::array<double, DIM> total = { 0.0, 0.0, 0.0 };
    for (const auto& force : forces)
    {
        for (int d = 0; d < DIM; ++d)
        {
            total[d] += force[d];
        }
    }
    return total;
}

std::vector<RVec> spreadForcesWithCpuVsite3(const std::vector<OpcVsiteGpuData>& sites,
                                            const std::vector<RVec>&            coordinates,
                                            const std::vector<RVec>&            initialForces)
{
    gmx_mtop_t mtop;
    mtop.moltype.emplace_back();
    gmx_moltype_t& moltype = mtop.moltype.back();
    init_t_atoms(&moltype.atoms, initialForces.size(), false);
    moltype.ilist = makeCpuVsiteLists(sites);

    std::vector<ParticleType> particleTypes(initialForces.size(), ParticleType::Atom);
    for (const auto& site : sites)
    {
        particleTypes[site.virtualSite]                 = ParticleType::VSite;
        moltype.atoms.atom[site.virtualSite].ptype      = ParticleType::VSite;
    }
    mtop.molblock.push_back({ 0, 1, {}, {} });
    mtop.natoms = initialForces.size();
    t_iparams parameter = {};
    parameter.vsite     = { 0.1477224_real, 0.1477224_real };
    mtop.ffparams.iparams.push_back(parameter);

    gmx_omp_nthreads_set(ModuleMultiThread::VirtualSite, 1);
    auto cpuVsites = makeVirtualSitesHandler(mtop, nullptr, PbcType::No, {});
    cpuVsites->setVirtualSites(&moltype.ilist,
                               initialForces.size(),
                               initialForces.size(),
                               particleTypes);
    std::vector<RVec> forces = initialForces;
    matrix            box    = { { 0 } };
    matrix            virial = { { 0 } };
    t_nrnb             nrnb   = {};
    cpuVsites->spreadForces(coordinates,
                            forces,
                            VirtualSitesHandler::VirialHandling::None,
                            {},
                            virial,
                            &nrnb,
                            box,
                            nullptr);
    return forces;
}

void constructPositionsWithCpuVsite3Pbc(const std::vector<OpcVsiteGpuData>& sites,
                                        std::vector<RVec>*                  coordinates,
                                        const matrix                       box)
{
    t_pbc pbc;
    set_pbc(&pbc, PbcType::Xyz, box);
    for (const auto& site : sites)
    {
        const RVec oldVirtualSite = (*coordinates)[site.virtualSite];
        RVec oxygenToHydrogen1;
        RVec oxygenToHydrogen2;
        pbc_dx_aiuc(&pbc,
                    (*coordinates)[site.hydrogen1],
                    (*coordinates)[site.oxygen],
                    oxygenToHydrogen1);
        pbc_dx_aiuc(&pbc,
                    (*coordinates)[site.hydrogen2],
                    (*coordinates)[site.oxygen],
                    oxygenToHydrogen2);
        const RVec constructedVirtualSite =
                (*coordinates)[site.oxygen] + site.weightH1 * oxygenToHydrogen1
                + site.weightH2 * oxygenToHydrogen2;
        RVec oldToConstructed;
        pbc_dx_aiuc(&pbc, constructedVirtualSite, oldVirtualSite, oldToConstructed);
        (*coordinates)[site.virtualSite] = oldVirtualSite + oldToConstructed;
    }
}

TEST(OpcVsiteGpuTest, PositionConstructionMatchesCpuVsite3)
{
    const auto& devices = getTestHardwareEnvironment()->getTestDeviceList();
    if (devices.empty())
    {
        GTEST_SKIP() << "No compatible CUDA GPU available";
    }

    const auto sites = threeOpcSites();
    auto       cpuCoordinates = testCoordinates();
    auto       gpuCoordinates = cpuCoordinates;
    const auto cpuLists       = makeCpuVsiteLists(sites);
    t_iparams  parameter      = {};
    parameter.vsite           = { 0.1477224_real, 0.1477224_real };
    const std::vector<t_iparams> parameters = { parameter };
    constructVirtualSites(cpuCoordinates, parameters, &cpuLists);

    for (const auto& device : devices)
    {
        const DeviceContext& context = device->deviceContext();
        const DeviceStream&  stream  = device->deviceStream();
        context.activate();
        DeviceBuffer<Float3> d_coordinates;
        allocateDeviceBuffer(&d_coordinates, gpuCoordinates.size(), context);
        copyToDeviceBuffer(&d_coordinates,
                           asGenericFloat3Pointer(gpuCoordinates.data()),
                           0,
                           gpuCoordinates.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        {
            OpcVsiteGpu opcVsites(sites, context, stream);
            matrix box = { { 10.0_real, 0, 0 }, { 0, 10.0_real, 0 }, { 0, 0, 10.0_real } };
            PbcAiuc pbcAiuc;
            setPbcAiuc(3, box, &pbcAiuc);
            opcVsites.constructPositions(d_coordinates, pbcAiuc);
            copyFromDeviceBuffer(asGenericFloat3Pointer(gpuCoordinates.data()),
                                 &d_coordinates,
                                 0,
                                 gpuCoordinates.size(),
                                 stream,
                                 GpuApiCallBehavior::Sync,
                                 nullptr);
        }
        freeDeviceBuffer(&d_coordinates);
        expectVectorsNear(gpuCoordinates, cpuCoordinates);
    }
}

TEST(OpcVsiteGpuTest, ForceSpreadMatchesCpuVsite3AndConservesTotalForce)
{
    const auto& devices = getTestHardwareEnvironment()->getTestDeviceList();
    if (devices.empty())
    {
        GTEST_SKIP() << "No compatible CUDA GPU available";
    }

    const auto sites = threeOpcSites();
    std::vector<RVec> initialForces(12);
    for (std::size_t atom = 0; atom < initialForces.size(); ++atom)
    {
        initialForces[atom] = { 0.25F * static_cast<float>(atom + 1),
                                -0.125F * static_cast<float>(atom + 2),
                                0.0625F * static_cast<float>(atom + 3) };
    }
    const auto expectedForces = spreadForcesWithCpuVsite3(sites, testCoordinates(), initialForces);
    const auto totalBefore = totalForce(initialForces);

    for (const auto& device : devices)
    {
        auto                 gpuForces = initialForces;
        const DeviceContext& context   = device->deviceContext();
        const DeviceStream&  stream    = device->deviceStream();
        context.activate();
        DeviceBuffer<Float3> d_forces;
        allocateDeviceBuffer(&d_forces, gpuForces.size(), context);
        copyToDeviceBuffer(&d_forces,
                           asGenericFloat3Pointer(gpuForces.data()),
                           0,
                           gpuForces.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        {
            OpcVsiteGpu opcVsites(sites, context, stream);
            opcVsites.spreadForces(d_forces);
            copyFromDeviceBuffer(asGenericFloat3Pointer(gpuForces.data()),
                                 &d_forces,
                                 0,
                                 gpuForces.size(),
                                 stream,
                                 GpuApiCallBehavior::Sync,
                                 nullptr);
        }
        freeDeviceBuffer(&d_forces);
        expectVectorsNear(gpuForces, expectedForces);
        const auto totalAfter = totalForce(gpuForces);
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(totalAfter[d], totalBefore[d], 1.0e-6);
        }
        for (const auto& site : sites)
        {
            EXPECT_FLOAT_EQ(gpuForces[site.virtualSite][XX], 0.0F);
            EXPECT_FLOAT_EQ(gpuForces[site.virtualSite][YY], 0.0F);
            EXPECT_FLOAT_EQ(gpuForces[site.virtualSite][ZZ], 0.0F);
        }
    }
}

TEST(OpcVsiteGpuTest, ForceSpreadFusedIntoForceReductionMatchesCpuReference)
{
    const auto& devices = getTestHardwareEnvironment()->getTestDeviceList();
    if (devices.empty())
    {
        GTEST_SKIP() << "No compatible CUDA GPU available";
    }

    const auto sites = threeOpcSites();
    std::vector<RVec> nbnxmForces(12);
    std::vector<RVec> rvecForces(12);
    std::vector<int>  cell(12);
    for (int atom = 0; atom < 12; ++atom)
    {
        nbnxmForces[atom] = { 0.25F * (atom + 1), -0.125F * (atom + 2), 0.0625F * (atom + 3) };
        rvecForces[atom]  = { -0.03125F * (atom + 3), 0.015625F * (atom + 1), 0.0078125F * atom };
        cell[atom]        = atom;
    }
    std::vector<RVec> combinedForces(12);
    for (int atom = 0; atom < 12; ++atom)
    {
        combinedForces[atom] = nbnxmForces[atom] + rvecForces[atom];
    }
    const auto expectedForces =
            spreadForcesWithCpuVsite3(sites, testCoordinates(), combinedForces);
    const OpcVsiteForceReductionFusion fusion = { 0, 3, 0.1477224F, 0.1477224F };

    for (const auto& device : devices)
    {
        auto outputForces = combinedForces;
        const DeviceContext& context = device->deviceContext();
        const DeviceStream&  stream  = device->deviceStream();
        context.activate();
        DeviceBuffer<Float3> d_nbnxmForces;
        DeviceBuffer<Float3> d_rvecForces;
        DeviceBuffer<Float3> d_outputForces;
        DeviceBuffer<int>    d_cell;
        allocateDeviceBuffer(&d_nbnxmForces, nbnxmForces.size(), context);
        allocateDeviceBuffer(&d_rvecForces, rvecForces.size(), context);
        allocateDeviceBuffer(&d_outputForces, outputForces.size(), context);
        allocateDeviceBuffer(&d_cell, cell.size(), context);
        copyToDeviceBuffer(&d_nbnxmForces,
                           asGenericFloat3Pointer(nbnxmForces.data()),
                           0,
                           nbnxmForces.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        copyToDeviceBuffer(&d_rvecForces,
                           asGenericFloat3Pointer(rvecForces.data()),
                           0,
                           rvecForces.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        copyToDeviceBuffer(&d_cell,
                           cell.data(),
                           0,
                           cell.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        launchForceReductionKernel(12,
                                   0,
                                   true,
                                   false,
                                   d_nbnxmForces,
                                   d_rvecForces,
                                   d_outputForces,
                                   d_cell,
                                   stream,
                                   nullptr,
                                   0,
                                   fusion);
        copyFromDeviceBuffer(asGenericFloat3Pointer(outputForces.data()),
                             &d_outputForces,
                             0,
                             outputForces.size(),
                             stream,
                             GpuApiCallBehavior::Sync,
                             nullptr);
        freeDeviceBuffer(&d_nbnxmForces);
        freeDeviceBuffer(&d_rvecForces);
        freeDeviceBuffer(&d_outputForces);
        freeDeviceBuffer(&d_cell);
        expectVectorsNear(outputForces, expectedForces);
    }
}

TEST(OpcVsiteGpuTest, PositionConstructionAcrossPbcMatchesCpuVsite3)
{
    const auto& devices = getTestHardwareEnvironment()->getTestDeviceList();
    if (devices.empty())
    {
        GTEST_SKIP() << "No compatible CUDA GPU available";
    }

    const std::vector<OpcVsiteGpuData> sites = { { 0, 1, 2, 3, 0.1477224F, 0.1477224F } };
    std::vector<RVec> cpuCoordinates = { { 9.98F, 4.00F, 4.00F },
                                         { 0.07F, 4.00F, 4.00F },
                                         { 9.95F, 4.08F, 4.00F },
                                         { 9.99F, 4.01F, 4.00F } };
    auto gpuCoordinates = cpuCoordinates;
    matrix box = { { 10.0_real, 0, 0 }, { 1.0_real, 9.0_real, 0 }, { 0.5_real, 0.25_real, 8.0_real } };
    constructPositionsWithCpuVsite3Pbc(sites, &cpuCoordinates, box);

    for (const auto& device : devices)
    {
        const DeviceContext& context = device->deviceContext();
        const DeviceStream&  stream  = device->deviceStream();
        context.activate();
        DeviceBuffer<Float3> d_coordinates;
        allocateDeviceBuffer(&d_coordinates, gpuCoordinates.size(), context);
        copyToDeviceBuffer(&d_coordinates,
                           asGenericFloat3Pointer(gpuCoordinates.data()),
                           0,
                           gpuCoordinates.size(),
                           stream,
                           GpuApiCallBehavior::Async,
                           nullptr);
        {
            PbcAiuc pbcAiuc;
            setPbcAiuc(3, box, &pbcAiuc);
            OpcVsiteGpu opcVsites(sites, context, stream);
            opcVsites.constructPositions(d_coordinates, pbcAiuc);
            copyFromDeviceBuffer(asGenericFloat3Pointer(gpuCoordinates.data()),
                                 &d_coordinates,
                                 0,
                                 gpuCoordinates.size(),
                                 stream,
                                 GpuApiCallBehavior::Sync,
                                 nullptr);
        }
        freeDeviceBuffer(&d_coordinates);
        expectVectorsNear(gpuCoordinates, cpuCoordinates);
    }
}

} // namespace
} // namespace gmx::test
