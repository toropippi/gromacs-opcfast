/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#include "gmxpre.h"

#include "gromacs/mdlib/opc_vsite_support.h"

#include <string>

#include <gtest/gtest.h>

#include "gromacs/topology/atoms.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"

namespace gmx::test
{
namespace
{

void makeOpcTopology(gmx_mtop_t* mtop, const int numWaters)
{
    mtop->moltype.emplace_back();
    gmx_moltype_t& moltype = mtop->moltype.back();
    init_t_atoms(&moltype.atoms, 4, false);

    moltype.atoms.atom[0].m     = 16.0_real;
    moltype.atoms.atom[1].m     = 1.008_real;
    moltype.atoms.atom[2].m     = 1.008_real;
    moltype.atoms.atom[3].m     = 0.0_real;
    moltype.atoms.atom[0].ptype = ParticleType::Atom;
    moltype.atoms.atom[1].ptype = ParticleType::Atom;
    moltype.atoms.atom[2].ptype = ParticleType::Atom;
    moltype.atoms.atom[3].ptype = ParticleType::VSite;

    moltype.ilist[InteractionFunction::SETTLE].iatoms      = { 0, 0, 1, 2 };
    moltype.ilist[InteractionFunction::VirtualSite3].iatoms = { 1, 3, 0, 1, 2 };

    mtop->molblock.push_back({ 0, numWaters, {}, {} });
    mtop->natoms = 4 * numWaters;

    t_iparams settleParameters = {};
    settleParameters.settle    = { 0.08724331_real, 0.1371205_real };
    t_iparams vsiteParameters  = {};
    vsiteParameters.vsite      = { 0.1477224_real, 0.1477224_real };
    mtop->ffparams.iparams      = { settleParameters, vsiteParameters };
}

TEST(OpcVsiteSupportTest, DetectsSupportedOpcVirtualSitesAndExpandsMolecules)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 3);
    const OpcVsiteAnalysis result = analyzeOpcGpuVsites(mtop);

    ASSERT_TRUE(result.hasVirtualSites);
    ASSERT_TRUE(result.supported()) << result.rejectionReason;
    ASSERT_EQ(result.sites.size(), 3);
    EXPECT_EQ(result.totalVirtualSiteCount, 3);
    EXPECT_EQ(result.sites[0].oxygen, 0);
    EXPECT_EQ(result.sites[0].virtualSite, 3);
    EXPECT_EQ(result.sites[2].oxygen, 8);
    EXPECT_EQ(result.sites[2].hydrogen1, 9);
    EXPECT_EQ(result.sites[2].hydrogen2, 10);
    EXPECT_EQ(result.sites[2].virtualSite, 11);
    EXPECT_FLOAT_EQ(result.sites[1].weightH1, 0.1477224F);
    EXPECT_FLOAT_EQ(result.sites[1].weightH2, 0.1477224F);
}

TEST(OpcVsiteSupportTest, AcceptsTopologyWithoutVirtualSites)
{
    gmx_mtop_t mtop;
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.hasVirtualSites);
    EXPECT_TRUE(result.supported());
    EXPECT_TRUE(result.sites.empty());
}

TEST(OpcVsiteSupportTest, DetectsOnlyOneContiguousOpcSiteBlockForForceReductionFusion)
{
    const std::vector<OpcVsiteGpuData> contiguous = {
        { 16, 17, 18, 19, 0.1477224F, 0.1477224F },
        { 20, 21, 22, 23, 0.1477224F, 0.1477224F }
    };
    EXPECT_TRUE(isContiguousOpcVsiteBlock(contiguous));

    auto separated = contiguous;
    separated[1]   = { 24, 25, 26, 27, 0.1477224F, 0.1477224F };
    EXPECT_FALSE(isContiguousOpcVsiteBlock(separated));

    auto differentWeights       = contiguous;
    differentWeights[1].weightH1 = 0.1477223F;
    EXPECT_FALSE(isContiguousOpcVsiteBlock(differentWeights));
    EXPECT_FALSE(isContiguousOpcVsiteBlock({}));
}

TEST(OpcVsiteSupportTest, RejectsNonOpcWeight)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 1);
    mtop.ffparams.iparams[1].vsite.a = 0.128_real;
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.supported());
    EXPECT_NE(result.rejectionReason.find("0.1477224"), std::string::npos);
}

TEST(OpcVsiteSupportTest, RejectsUnsupportedVirtualSiteFunction)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 1);
    mtop.moltype[0].ilist[InteractionFunction::VirtualSite3].iatoms.clear();
    mtop.moltype[0].ilist[InteractionFunction::VirtualSite3FlexibleDistance].iatoms = { 1, 3, 0, 1, 2 };
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.supported());
    EXPECT_NE(result.rejectionReason.find("VSITE3 function 1"), std::string::npos);
}

TEST(OpcVsiteSupportTest, RejectsWaterWithoutSettle)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 1);
    mtop.moltype[0].ilist[InteractionFunction::SETTLE].iatoms.clear();
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.supported());
    EXPECT_NE(result.rejectionReason.find("SETTLE"), std::string::npos);
}

TEST(OpcVsiteSupportTest, RejectsNonContiguousAtomOrder)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 1);
    mtop.moltype[0].ilist[InteractionFunction::VirtualSite3].iatoms = { 1, 3, 1, 0, 2 };
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.supported());
    EXPECT_NE(result.rejectionReason.find("O/H1/H2/EPW"), std::string::npos);
}

TEST(OpcVsiteSupportTest, RejectsNonMasslessVirtualSite)
{
    gmx_mtop_t mtop;
    makeOpcTopology(&mtop, 1);
    mtop.moltype[0].atoms.atom[3].m = 1.0_real;
    const auto result = analyzeOpcGpuVsites(mtop);
    EXPECT_FALSE(result.supported());
    EXPECT_NE(result.rejectionReason.find("massless"), std::string::npos);
}

} // namespace
} // namespace gmx::test
