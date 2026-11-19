/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#ifndef GMX_MDLIB_OPC_VSITE_SUPPORT_H
#define GMX_MDLIB_OPC_VSITE_SUPPORT_H

#include <string>
#include <vector>

#include "gromacs/utility/arrayref.h"

struct gmx_mtop_t;

namespace gmx
{

//! Indices and linear VSITE3 weights for one strictly validated OPC water.
struct OpcVsiteGpuData
{
    int   oxygen;
    int   hydrogen1;
    int   hydrogen2;
    int   virtualSite;
    float weightH1;
    float weightH2;
};

//! Result of strict topology validation for the CUDA OPC virtual-site MVP.
struct OpcVsiteAnalysis
{
    bool                         hasVirtualSites       = false;
    int                          totalVirtualSiteCount = 0;
    std::vector<OpcVsiteGpuData> sites;
    std::string                  rejectionReason;

    //! Whether every virtual site in the topology is supported by the MVP.
    bool supported() const { return rejectionReason.empty(); }
};

/*! \brief Detect strictly supported OPC VSITE3 waters in a global topology.
 *
 * The supported layout is exactly O/H1/H2/EPW in a four-atom molecule, one
 * SETTLE over O/H1/H2, one linear VSITE3 function 1 for EPW, OPC weights
 * a=b=0.1477224, and a massless virtual-site particle. This strict contract
 * makes constructing atoms unique and keeps every construction within one
 * SETTLE update group, so the CUDA force-spreading kernel needs no atomics.
 * Position construction still applies molecular PBC and preserves the prior
 * EPW periodic image, matching the existing CPU virtual-site implementation.
 */
OpcVsiteAnalysis analyzeOpcGpuVsites(const gmx_mtop_t& mtop);

//! Whether all validated sites form one arithmetic O/H1/H2/EPW block with common weights.
bool isContiguousOpcVsiteBlock(ArrayRef<const OpcVsiteGpuData> sites);

} // namespace gmx
#endif
