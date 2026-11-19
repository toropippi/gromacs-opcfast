/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#include "gmxpre.h"

#include "opc_vsite_gpu.h"

#include <cstdio>

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/mdlib/opc_vsite_gpu_internal.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

OpcVsiteGpu::OpcVsiteGpu(const std::vector<OpcVsiteGpuData>& sites,
                         const DeviceContext&                deviceContext,
                         const DeviceStream&                 deviceStream) :
    deviceContext_(deviceContext), deviceStream_(deviceStream), numSites_(sites.size())
{
    static_assert(sizeof(real) == sizeof(float), "OPC CUDA virtual sites require mixed precision");
    if (numSites_ > 0)
    {
        allocateDeviceBuffer(&d_sites_, numSites_, deviceContext_);
        copyToDeviceBuffer(&d_sites_,
                           sites.data(),
                           0,
                           numSites_,
                           deviceStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }
}

OpcVsiteGpu::~OpcVsiteGpu()
{
    if (numSites_ == 0)
    {
        return;
    }
    try
    {
        deviceStream_.synchronize();
        freeDeviceBuffer(&d_sites_);
    }
    catch (const InternalError& error)
    {
        std::fprintf(stderr, "Internal error in OpcVsiteGpu destructor: %s\n", error.what());
    }
}

void OpcVsiteGpu::constructPositions(DeviceBuffer<Float3> coordinates, const PbcAiuc& pbcAiuc) const
{
    if (numSites_ > 0)
    {
        launchOpcVsiteConstructionKernel(numSites_, d_sites_, coordinates, pbcAiuc, deviceStream_);
    }
}

void OpcVsiteGpu::constructVelocities(DeviceBuffer<Float3> velocities) const
{
    if (numSites_ > 0)
    {
        launchOpcVsiteLinearConstructionKernel(numSites_, d_sites_, velocities, deviceStream_);
    }
}

void OpcVsiteGpu::spreadForces(DeviceBuffer<Float3> forces) const
{
    if (numSites_ > 0)
    {
        launchOpcVsiteSpreadKernel(numSites_, d_sites_, forces, deviceStream_);
    }
}

} // namespace gmx
