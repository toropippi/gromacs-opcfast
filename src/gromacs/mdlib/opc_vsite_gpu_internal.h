/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#ifndef GMX_MDLIB_OPC_VSITE_GPU_INTERNAL_H
#define GMX_MDLIB_OPC_VSITE_GPU_INTERNAL_H

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/mdlib/opc_vsite_support.h"
#include "gromacs/pbcutil/pbc_aiuc.h"

namespace gmx
{

void launchOpcVsiteConstructionKernel(int                                      numSites,
                                      const DeviceBuffer<OpcVsiteGpuData>&      sites,
                                      DeviceBuffer<Float3>                      coordinates,
                                      const PbcAiuc&                            pbcAiuc,
                                      const DeviceStream&                       deviceStream);

void launchOpcVsiteLinearConstructionKernel(int                                 numSites,
                                            const DeviceBuffer<OpcVsiteGpuData>& sites,
                                            DeviceBuffer<Float3>                 values,
                                            const DeviceStream&                  deviceStream);

void launchOpcVsiteSpreadKernel(int                                      numSites,
                                const DeviceBuffer<OpcVsiteGpuData>&      sites,
                                DeviceBuffer<Float3>                      forces,
                                const DeviceStream&                       deviceStream);

} // namespace gmx
#endif
