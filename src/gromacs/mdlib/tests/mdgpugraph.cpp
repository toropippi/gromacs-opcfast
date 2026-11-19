/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
 * \brief Tests for MD GPU graph
 *
 * \author Alan Gray <alang@nvidia.com>
 */
#include "gmxpre.h"

#include "config.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#if GMX_HAVE_GPU_GRAPH_SUPPORT

#    include "gromacs/gpu_utils/device_stream.h"
#    include "gromacs/gpu_utils/device_stream_manager.h"
#    include "gromacs/gpu_utils/devicebuffer.h"
#    include "gromacs/gpu_utils/gpueventsynchronizer.h"
#    include "gromacs/gpu_utils/hostallocator.h"
#    include "gromacs/mdlib/mdgraph_gpu.h"
#    include "gromacs/mdlib/opc_vsite_gpu.h"
#    include "gromacs/mdlib/opc_vsite_support.h"
#    include "gromacs/pbcutil/pbc_aiuc.h"
#    include "gromacs/utility/vectypes.h"

#    include "testutils/refdata.h"
#    include "testutils/test_hardware_environment.h"
#    include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Get simulation workload structure with appropriate components for these tests.
 *
 * useMdGpuGraph is true to enable the graph, useGpuUpdate is true to
 * provide an update stream to use in the tests,
 * havePpDomainDecomposition is false since these are single-rank
 * tests, and useGpuPme is false since interoperation with the PME
 * stream relies on an implicit sync in mdrun.
 * \returns simulationWork Simulation workload structure
 */
SimulationWorkload makeSimulationWorkForMdGpuGraph()
{
    SimulationWorkload simulationWork;
    simulationWork.useMdGpuGraph             = true;
    simulationWork.useGpuUpdate              = true;
    simulationWork.havePpDomainDecomposition = false;
    simulationWork.useGpuPme                 = false;
    simulationWork.haveSeparatePmeRank       = false;
    return simulationWork;
}

TEST(MdGraphTest, MdGpuGraphExecutesActivities)
{

    const auto& testDeviceList = getTestHardwareEnvironment()->getTestDeviceList();
    if (testDeviceList.empty())
    {
        GTEST_SKIP() << "No compatible GPUs to test on.";
    }
    for (const auto& testDevice : testDeviceList)
    {
#    if GMX_GPU_SYCL
        if (!testDevice->deviceInfo().supportsSyclGraph)
        {
            GTEST_SKIP() << "Device does not support GPU Graphs";
        }
#    endif
        const auto& deviceContext = testDevice->deviceContext();

        // Initialize required structures
        SimulationWorkload   simulationWork = makeSimulationWorkForMdGpuGraph();
        DeviceStreamManager  deviceStreamManager(testDevice->deviceInfo(), simulationWork, false);
        GpuEventSynchronizer xReadyOnDeviceEvent;
        GpuEventSynchronizer xUpdatedOnDeviceEvent;
        gmx::MdGpuGraph      mdGpuGraph(
                deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
        gmx::MdGpuGraph mdGpuGraphAlternate(
                deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
        mdGpuGraph.setAlternateStepPpTaskCompletionEvent(mdGpuGraphAlternate.getPpTaskCompletionEvent());

        // Allocate 2 device buffers
        DeviceBuffer<int> d_output;
        int               d_output_size       = -1;
        int               d_output_size_alloc = -1;
        reallocateDeviceBuffer(&d_output, 1, &d_output_size, &d_output_size_alloc, deviceContext);
        DeviceBuffer<int> d_staging;
        int               d_staging_size       = -1;
        int               d_staging_size_alloc = -1;
        reallocateDeviceBuffer(&d_staging, 1, &d_staging_size, &d_staging_size_alloc, deviceContext);

        // Perform below steps with and without graph
        for (bool useGraph : { false, true })
        {

            HostVector<int> h_one;
            changePinningPolicy(&h_one, PinningPolicy::PinnedIfSupported);
            h_one.resize(1);
            h_one[0] = 1;

            HostVector<int> h_output;
            changePinningPolicy(&h_output, PinningPolicy::PinnedIfSupported);
            h_output.resize(1);

            // Set output to 1 on GPU
            copyToDeviceBuffer(&d_output,
                               h_one.data(),
                               0,
                               1,
                               deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                               GpuApiCallBehavior::Sync,
                               nullptr);

            if (useGraph && mdGpuGraph.captureThisStep(true)) // denote start of graph region
            {
                // Start graph capture (automatically on local stream)
                bool usedGraphLastStep = true;
                mdGpuGraph.setUsedGraphLastStep(usedGraphLastStep);
                mdGpuGraph.startRecord(&xReadyOnDeviceEvent);
            }

            // Clear output on GPU in update stream, which will be automatically forked from
            // local stream in the graph. Can be done in single call, but instead perform in
            // 2 stages to create a 2-node graph.
            const DeviceStream& stream =
                    deviceStreamManager.stream(gmx::DeviceStreamType::UpdateAndConstraints);
            clearDeviceBufferAsync(&d_staging, 0, 1, stream);
            copyBetweenDeviceBuffers(&d_output, &d_staging, 1, stream, GpuApiCallBehavior::Async, nullptr);

            if (mdGpuGraph.graphIsCapturingThisStep()) // denote end of graph region
            {
                // End graph capture and instantiate
                mdGpuGraph.endRecord();
                bool forceGraphReinstantiation = true;
                mdGpuGraph.createExecutableGraph(forceGraphReinstantiation);
            }

            // Wait for update stream
            deviceStreamManager.stream(gmx::DeviceStreamType::UpdateAndConstraints).synchronize();
            // Synchronously copy output to host buffer
            copyFromDeviceBuffer(h_output.data(),
                                 &d_output,
                                 0,
                                 1,
                                 deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                                 GpuApiCallBehavior::Sync,
                                 nullptr);

            if (mdGpuGraph.useGraphThisStep())
            {
                // Graph has not yet been launched so output has not yet been cleared
                EXPECT_EQ(h_output[0], 1);
            }
            else
            {
                // Without graph capture active, the above memory operations will have been performed directly and output will now be cleared
                EXPECT_EQ(h_output[0], 0);
            }

            if (mdGpuGraph.useGraphThisStep())
            {
                // Now launch graph and check output is cleared
                mdGpuGraph.launchGraphMdStep(&xUpdatedOnDeviceEvent);
                xUpdatedOnDeviceEvent.waitForEvent();

                // Synchronously copy output to host buffer in local stream
                copyFromDeviceBuffer(h_output.data(),
                                     &d_output,
                                     0,
                                     1,
                                     deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                                     GpuApiCallBehavior::Sync,
                                     nullptr);

                // Graph has now been executed, so output is cleared
                EXPECT_EQ(h_output[0], 0);
            }
        }
    }
}

TEST(MdGraphTest, OpcVsiteKernelsCaptureAndReplay)
{
    const auto& testDeviceList = getTestHardwareEnvironment()->getTestDeviceList();
    if (testDeviceList.empty())
    {
        GTEST_SKIP() << "No compatible GPUs to test on.";
    }

    constexpr float weightH = 0.1477224F;
    const std::vector<OpcVsiteGpuData> sites = { { 0, 1, 2, 3, weightH, weightH } };
    const std::array<Float3, 4> initialCoordinates = {
        Float3{ 1.0F, 1.0F, 1.0F },
        Float3{ 1.1F, 1.0F, 1.0F },
        Float3{ 1.0F, 1.1F, 1.0F },
        Float3{ 1.0F, 1.0F, 1.0F }
    };
    const std::array<Float3, 4> initialVelocities = {
        Float3{ 1.0F, 2.0F, 3.0F },
        Float3{ 2.0F, 3.0F, 4.0F },
        Float3{ 3.0F, 4.0F, 5.0F },
        Float3{ 0.0F, 0.0F, 0.0F }
    };
    const std::array<Float3, 4> initialForces = {
        Float3{ 1.0F, 2.0F, 3.0F },
        Float3{ 4.0F, 5.0F, 6.0F },
        Float3{ 7.0F, 8.0F, 9.0F },
        Float3{ 10.0F, 20.0F, 30.0F }
    };

    for (const auto& testDevice : testDeviceList)
    {
#    if GMX_GPU_SYCL
        if (!testDevice->deviceInfo().supportsSyclGraph)
        {
            GTEST_SKIP() << "Device does not support GPU Graphs";
        }
#    endif
        const DeviceContext& deviceContext = testDevice->deviceContext();
        deviceContext.activate();
        SimulationWorkload  simulationWork = makeSimulationWorkForMdGpuGraph();
        DeviceStreamManager deviceStreamManager(testDevice->deviceInfo(), simulationWork, false);
        const DeviceStream&  localStream =
                deviceStreamManager.stream(DeviceStreamType::NonBondedLocal);
        const DeviceStream& updateStream =
                deviceStreamManager.stream(DeviceStreamType::UpdateAndConstraints);

        DeviceBuffer<Float3> d_coordinates;
        DeviceBuffer<Float3> d_velocities;
        DeviceBuffer<Float3> d_forces;
        allocateDeviceBuffer(&d_coordinates, initialCoordinates.size(), deviceContext);
        allocateDeviceBuffer(&d_velocities, initialVelocities.size(), deviceContext);
        allocateDeviceBuffer(&d_forces, initialForces.size(), deviceContext);
        copyToDeviceBuffer(&d_coordinates,
                           initialCoordinates.data(),
                           0,
                           initialCoordinates.size(),
                           updateStream,
                           GpuApiCallBehavior::Sync,
                           nullptr);
        copyToDeviceBuffer(&d_velocities,
                           initialVelocities.data(),
                           0,
                           initialVelocities.size(),
                           updateStream,
                           GpuApiCallBehavior::Sync,
                           nullptr);
        copyToDeviceBuffer(&d_forces,
                           initialForces.data(),
                           0,
                           initialForces.size(),
                           localStream,
                           GpuApiCallBehavior::Sync,
                           nullptr);

        OpcVsiteGpu positionVsites(sites, deviceContext, updateStream);
        OpcVsiteGpu forceVsites(sites, deviceContext, localStream);
        GpuEventSynchronizer xReadyOnDeviceEvent;
        GpuEventSynchronizer xUpdatedOnDeviceEvent;
        MdGpuGraph mdGpuGraph(deviceStreamManager,
                              simulationWork,
                              MPI_COMM_WORLD,
                              MdGraphEvenOrOddStep::EvenStep,
                              nullptr);
        MdGpuGraph mdGpuGraphAlternate(deviceStreamManager,
                                       simulationWork,
                                       MPI_COMM_WORLD,
                                       MdGraphEvenOrOddStep::OddStep,
                                       nullptr);
        mdGpuGraph.setAlternateStepPpTaskCompletionEvent(
                mdGpuGraphAlternate.getPpTaskCompletionEvent());

        xReadyOnDeviceEvent.markEvent(localStream);
        ASSERT_TRUE(mdGpuGraph.captureThisStep(true));
        mdGpuGraph.setUsedGraphLastStep(false);
        mdGpuGraph.startRecord(&xReadyOnDeviceEvent);

        matrix box = { { 10.0_real, 0, 0 }, { 0, 10.0_real, 0 }, { 0, 0, 10.0_real } };
        PbcAiuc pbcAiuc;
        setPbcAiuc(3, box, &pbcAiuc);
        forceVsites.spreadForces(d_forces);
        positionVsites.constructPositions(d_coordinates, pbcAiuc);
        positionVsites.constructVelocities(d_velocities);

        mdGpuGraph.endRecord();
        mdGpuGraph.createExecutableGraph(true);
        mdGpuGraph.launchGraphMdStep(&xUpdatedOnDeviceEvent);
        xUpdatedOnDeviceEvent.waitForEvent();

        std::array<Float3, 4> coordinates;
        std::array<Float3, 4> velocities;
        std::array<Float3, 4> forces;
        copyFromDeviceBuffer(coordinates.data(),
                             &d_coordinates,
                             0,
                             coordinates.size(),
                             updateStream,
                             GpuApiCallBehavior::Sync,
                             nullptr);
        copyFromDeviceBuffer(velocities.data(),
                             &d_velocities,
                             0,
                             velocities.size(),
                             updateStream,
                             GpuApiCallBehavior::Sync,
                             nullptr);
        copyFromDeviceBuffer(forces.data(),
                             &d_forces,
                             0,
                             forces.size(),
                             localStream,
                             GpuApiCallBehavior::Sync,
                             nullptr);

        EXPECT_NEAR(coordinates[3][XX], 1.0F + 0.1F * weightH, 2.0e-6F);
        EXPECT_NEAR(coordinates[3][YY], 1.0F + 0.1F * weightH, 2.0e-6F);
        EXPECT_NEAR(coordinates[3][ZZ], 1.0F, 2.0e-6F);
        const float weightO = 1.0F - 2.0F * weightH;
        for (int d = 0; d < DIM; ++d)
        {
            const float expectedVelocity = weightO * initialVelocities[0][d]
                                           + weightH * initialVelocities[1][d]
                                           + weightH * initialVelocities[2][d];
            EXPECT_NEAR(velocities[3][d], expectedVelocity, 2.0e-6F);
            EXPECT_NEAR(forces[0][d], initialForces[0][d] + weightO * initialForces[3][d], 2.0e-6F);
            EXPECT_NEAR(forces[1][d], initialForces[1][d] + weightH * initialForces[3][d], 2.0e-6F);
            EXPECT_NEAR(forces[2][d], initialForces[2][d] + weightH * initialForces[3][d], 2.0e-6F);
            EXPECT_FLOAT_EQ(forces[3][d], 0.0F);
        }

        freeDeviceBuffer(&d_coordinates);
        freeDeviceBuffer(&d_velocities);
        freeDeviceBuffer(&d_forces);
    }
}

TEST(MdGraphTest, MdGpuGraphCaptureAndUsageConsistency)
{

    const auto& testDeviceList = getTestHardwareEnvironment()->getTestDeviceList();
    if (testDeviceList.empty())
    {
        GTEST_SKIP() << "No compatible GPUs to test on.";
    }
    for (const auto& testDevice : testDeviceList)
    {
#    if GMX_GPU_SYCL
        if (!testDevice->deviceInfo().supportsSyclGraph)
        {
            GTEST_SKIP() << "Device does not support GPU Graphs";
        }
#    endif
        const auto& deviceContext = testDevice->deviceContext();

        // Initialize required structures
        SimulationWorkload   simulationWork = makeSimulationWorkForMdGpuGraph();
        DeviceStreamManager  deviceStreamManager(testDevice->deviceInfo(), simulationWork, false);
        GpuEventSynchronizer xReadyOnDeviceEvent;
        GpuEventSynchronizer xUpdatedOnDeviceEvent;
        gmx::MdGpuGraph      mdGpuGraph(
                deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
        gmx::MdGpuGraph mdGpuGraphAlternate(
                deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
        mdGpuGraph.setAlternateStepPpTaskCompletionEvent(mdGpuGraphAlternate.getPpTaskCompletionEvent());

        DeviceBuffer<int> d_buf;
        int               d_buf_size       = -1;
        int               d_buf_size_alloc = -1;
        reallocateDeviceBuffer(&d_buf, 1, &d_buf_size, &d_buf_size_alloc, deviceContext);

        bool usedGraphLastStep = false;
        mdGpuGraph.setUsedGraphLastStep(usedGraphLastStep);

        // Step 1: don't use graph
        bool canUseGraphThisStep = false;
        ASSERT_FALSE(mdGpuGraph.captureThisStep(canUseGraphThisStep));
        ASSERT_FALSE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_FALSE(mdGpuGraph.useGraphThisStep());

        // Step 2: capture graph
        canUseGraphThisStep = true;
        // No work in UpdateAndConstraints stream, the event is immediately ready
        xReadyOnDeviceEvent.markEvent(deviceStreamManager.stream(DeviceStreamType::UpdateAndConstraints));
        ASSERT_TRUE(mdGpuGraph.captureThisStep(canUseGraphThisStep));
        mdGpuGraph.startRecord(&xReadyOnDeviceEvent);
        ASSERT_TRUE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_TRUE(mdGpuGraph.useGraphThisStep());
        EXPECT_TRUE(xReadyOnDeviceEvent.isMarked());

        // Activity to capture in graph
        clearDeviceBufferAsync(
                &d_buf, 0, 1, deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal));

        // End graph capture and instantiate
        mdGpuGraph.endRecord();
        bool forceGraphReinstantiation = true;
        mdGpuGraph.createExecutableGraph(forceGraphReinstantiation);

        // Launch graph
        xUpdatedOnDeviceEvent.reset();
        mdGpuGraph.launchGraphMdStep(&xUpdatedOnDeviceEvent);
        EXPECT_TRUE(xUpdatedOnDeviceEvent.isMarked());

        // Step 3: re-use existing graph
        ASSERT_FALSE(mdGpuGraph.captureThisStep(canUseGraphThisStep));
        ASSERT_FALSE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_TRUE(mdGpuGraph.useGraphThisStep());

        // Launch graph
        xUpdatedOnDeviceEvent.reset();
        mdGpuGraph.launchGraphMdStep(&xUpdatedOnDeviceEvent);
        EXPECT_TRUE(xUpdatedOnDeviceEvent.isMarked());

        // Step 4: don't use graph, even although it exists
        canUseGraphThisStep = false;
        ASSERT_FALSE(mdGpuGraph.captureThisStep(canUseGraphThisStep));
        ASSERT_FALSE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_FALSE(mdGpuGraph.useGraphThisStep());

        // Step 5: re-use existing graph again
        canUseGraphThisStep = true;
        ASSERT_FALSE(mdGpuGraph.captureThisStep(canUseGraphThisStep));
        ASSERT_FALSE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_TRUE(mdGpuGraph.useGraphThisStep());

        // Step 6: reset graph
        mdGpuGraph.reset();
        ASSERT_FALSE(mdGpuGraph.graphIsCapturingThisStep());
        ASSERT_FALSE(mdGpuGraph.useGraphThisStep());
    }
}

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_HAVE_GPU_GRAPH_SUPPORT
