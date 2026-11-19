/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#include "gmxpre.h"

#include "opc_vsite_gpu_internal.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/typecasts_cuda_hip.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/utility/vec.h"

namespace gmx
{
namespace
{

constexpr int c_threadsPerBlock = 256;

__launch_bounds__(c_threadsPerBlock) __global__
        void constructOpcVsitesKernel(const int                         numSites,
                                     const OpcVsiteGpuData* __restrict__ sites,
                                     float3* __restrict__               coordinates,
                                     const PbcAiuc                      pbcAiuc)
{
    const int siteIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (siteIndex >= numSites)
    {
        return;
    }

    const OpcVsiteGpuData site = sites[siteIndex];
    const float3 oxygen        = coordinates[site.oxygen];
    const float3 hydrogen1     = coordinates[site.hydrogen1];
    const float3 hydrogen2     = coordinates[site.hydrogen2];
    const float3 oldVirtualSite = coordinates[site.virtualSite];
    const float3 oxygenToHydrogen1 = pbcDxAiuc(pbcAiuc, hydrogen1, oxygen);
    const float3 oxygenToHydrogen2 = pbcDxAiuc(pbcAiuc, hydrogen2, oxygen);

    const float3 constructedVirtualSite =
            make_float3(oxygen.x + site.weightH1 * oxygenToHydrogen1.x
                                 + site.weightH2 * oxygenToHydrogen2.x,
                        oxygen.y + site.weightH1 * oxygenToHydrogen1.y
                                 + site.weightH2 * oxygenToHydrogen2.y,
                        oxygen.z + site.weightH1 * oxygenToHydrogen1.z
                                 + site.weightH2 * oxygenToHydrogen2.z);

    // Match the CPU virtual-site implementation: after constructing with minimum-image
    // O-H vectors, retain the periodic image occupied by EPW before this update. This is
    // required by the GPU neighbor-search grid, whose atom-to-image mapping remains fixed
    // between pair-list rebuilds.
    const float3 oldToConstructed = pbcDxAiuc(pbcAiuc, constructedVirtualSite, oldVirtualSite);
    coordinates[site.virtualSite] = oldVirtualSite + oldToConstructed;
}

__launch_bounds__(c_threadsPerBlock) __global__
        void constructOpcVsiteLinearValuesKernel(const int                         numSites,
                                                const OpcVsiteGpuData* __restrict__ sites,
                                                float3* __restrict__               values)
{
    const int siteIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (siteIndex >= numSites)
    {
        return;
    }

    const OpcVsiteGpuData site = sites[siteIndex];
    const float3 oxygen        = values[site.oxygen];
    const float3 hydrogen1     = values[site.hydrogen1];
    const float3 hydrogen2     = values[site.hydrogen2];
    const float  weightO       = 1.0F - site.weightH1 - site.weightH2;

    values[site.virtualSite] = make_float3(weightO * oxygen.x + site.weightH1 * hydrogen1.x
                                                    + site.weightH2 * hydrogen2.x,
                                            weightO * oxygen.y + site.weightH1 * hydrogen1.y
                                                    + site.weightH2 * hydrogen2.y,
                                            weightO * oxygen.z + site.weightH1 * hydrogen1.z
                                                    + site.weightH2 * hydrogen2.z);
}

__launch_bounds__(c_threadsPerBlock) __global__
        void spreadOpcVsiteForcesKernel(const int                         numSites,
                                       const OpcVsiteGpuData* __restrict__ sites,
                                       float3* __restrict__               forces)
{
    const int siteIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (siteIndex >= numSites)
    {
        return;
    }

    // analyzeOpcGpuVsites guarantees unique O/H/EPW atoms for every entry.
    // Therefore these non-atomic updates cannot race.
    const OpcVsiteGpuData site    = sites[siteIndex];
    const float3          epForce = forces[site.virtualSite];
    const float           weightO = 1.0F - site.weightH1 - site.weightH2;

    float3 oxygenForce    = forces[site.oxygen];
    float3 hydrogen1Force = forces[site.hydrogen1];
    float3 hydrogen2Force = forces[site.hydrogen2];

    oxygenForce.x += weightO * epForce.x;
    oxygenForce.y += weightO * epForce.y;
    oxygenForce.z += weightO * epForce.z;
    hydrogen1Force.x += site.weightH1 * epForce.x;
    hydrogen1Force.y += site.weightH1 * epForce.y;
    hydrogen1Force.z += site.weightH1 * epForce.z;
    hydrogen2Force.x += site.weightH2 * epForce.x;
    hydrogen2Force.y += site.weightH2 * epForce.y;
    hydrogen2Force.z += site.weightH2 * epForce.z;

    forces[site.oxygen]     = oxygenForce;
    forces[site.hydrogen1]  = hydrogen1Force;
    forces[site.hydrogen2]  = hydrogen2Force;
    forces[site.virtualSite] = make_float3(0.0F, 0.0F, 0.0F);
}

KernelLaunchConfig launchConfig(const int numSites)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = divideRoundUp(numSites, c_threadsPerBlock);
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;
    return config;
}

} // namespace

void launchOpcVsiteConstructionKernel(const int                                numSites,
                                      const DeviceBuffer<OpcVsiteGpuData>&      sites,
                                      DeviceBuffer<Float3>                      coordinates,
                                      const PbcAiuc&                            pbcAiuc,
                                      const DeviceStream&                       deviceStream)
{
    const KernelLaunchConfig config = launchConfig(numSites);
    const auto arguments = prepareGpuKernelArguments(constructOpcVsitesKernel,
                                                     config,
                                                     &numSites,
                                                     &sites,
                                                     asFloat3Pointer(&coordinates),
                                                     &pbcAiuc);
    launchGpuKernel(constructOpcVsitesKernel,
                    config,
                    deviceStream,
                    nullptr,
                    "constructOpcVsitesKernel",
                    arguments);
}

void launchOpcVsiteLinearConstructionKernel(const int                                numSites,
                                            const DeviceBuffer<OpcVsiteGpuData>&      sites,
                                            DeviceBuffer<Float3>                      values,
                                            const DeviceStream&                       deviceStream)
{
    const KernelLaunchConfig config = launchConfig(numSites);
    const auto arguments = prepareGpuKernelArguments(constructOpcVsiteLinearValuesKernel,
                                                     config,
                                                     &numSites,
                                                     &sites,
                                                     asFloat3Pointer(&values));
    launchGpuKernel(constructOpcVsiteLinearValuesKernel,
                    config,
                    deviceStream,
                    nullptr,
                    "constructOpcVsiteLinearValuesKernel",
                    arguments);
}

void launchOpcVsiteSpreadKernel(const int                                numSites,
                                const DeviceBuffer<OpcVsiteGpuData>&      sites,
                                DeviceBuffer<Float3>                      forces,
                                const DeviceStream&                       deviceStream)
{
    const KernelLaunchConfig config = launchConfig(numSites);
    const auto arguments = prepareGpuKernelArguments(spreadOpcVsiteForcesKernel,
                                                     config,
                                                     &numSites,
                                                     &sites,
                                                     asFloat3Pointer(&forces));
    launchGpuKernel(spreadOpcVsiteForcesKernel,
                    config,
                    deviceStream,
                    nullptr,
                    "spreadOpcVsiteForcesKernel",
                    arguments);
}

} // namespace gmx
