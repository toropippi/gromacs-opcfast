/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#ifndef GMX_MDLIB_OPC_VSITE_GPU_H
#define GMX_MDLIB_OPC_VSITE_GPU_H

#include <vector>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/mdlib/opc_vsite_support.h"
#include "gromacs/pbcutil/pbc_aiuc.h"

namespace gmx
{

/*! \brief Owns device topology data and launches strict OPC VSITE3 kernels.
 *
 * The caller must provide data returned by analyzeOpcGpuVsites(). The strict
 * topology contract guarantees that no two sites write the same constructing
 * atom, so force spreading is race-free without atomics.
 */
class OpcVsiteGpu
{
public:
    OpcVsiteGpu(const std::vector<OpcVsiteGpuData>& sites,
                const DeviceContext&                deviceContext,
                const DeviceStream&                 deviceStream);
    ~OpcVsiteGpu();

    OpcVsiteGpu(const OpcVsiteGpu&)            = delete;
    OpcVsiteGpu& operator=(const OpcVsiteGpu&) = delete;

    //! Enqueue EPW construction after GPU update/SETTLE on this object's stream.
    void constructPositions(DeviceBuffer<Float3> coordinates, const PbcAiuc& pbcAiuc) const;

    //! Enqueue the same linear construction for virtual-site velocities.
    void constructVelocities(DeviceBuffer<Float3> velocities) const;

    //! Enqueue EPW force spreading and EPW force clearing before GPU update.
    void spreadForces(DeviceBuffer<Float3> forces) const;

    int numSites() const { return numSites_; }

private:
    const DeviceContext&              deviceContext_;
    const DeviceStream&               deviceStream_;
    int                               numSites_ = 0;
    DeviceBuffer<OpcVsiteGpuData>     d_sites_;
};

} // namespace gmx
#endif
