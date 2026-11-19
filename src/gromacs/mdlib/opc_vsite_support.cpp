/*
 * This file is part of the GROMACS molecular simulation package.
 * Copyright 2026 The GROMACS Authors.
 */

#include "gmxpre.h"

#include "opc_vsite_support.h"

#include <cmath>
#include <string>
#include <vector>

#include "gromacs/mdlib/vsite.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/topology/topology_enums.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{
namespace
{

constexpr real c_opcWeight       = 0.1477224_real;
constexpr real c_weightTolerance = 5.0e-7_real;

bool hasAnyVsiteInteraction(const InteractionLists& lists)
{
    for (const InteractionFunction functionType : vSiteFunctionTypes)
    {
        if (!lists[functionType].empty())
        {
            return true;
        }
    }
    return false;
}

std::string validateOpcMoleculeType(const gmx_mtop_t& mtop, const int moleculeType)
{
    const gmx_moltype_t& moltype = mtop.moltype[moleculeType];

    for (const InteractionFunction functionType : vSiteFunctionTypes)
    {
        if (functionType != InteractionFunction::VirtualSite3 && !moltype.ilist[functionType].empty())
        {
            return formatString("molecule type %d contains a virtual site other than linear VSITE3 function 1",
                                moleculeType);
        }
    }

    const auto& vsiteAtoms = moltype.ilist[InteractionFunction::VirtualSite3].iatoms;
    if (vsiteAtoms.size() != 5)
    {
        return formatString("molecule type %d must contain exactly one VSITE3 function 1", moleculeType);
    }
    if (moltype.atoms.nr != 4 || vsiteAtoms[1] != 3 || vsiteAtoms[2] != 0 || vsiteAtoms[3] != 1
        || vsiteAtoms[4] != 2)
    {
        return formatString("molecule type %d does not have contiguous O/H1/H2/EPW atom order", moleculeType);
    }

    const auto& settleAtoms = moltype.ilist[InteractionFunction::SETTLE].iatoms;
    if (settleAtoms.size() != 4 || settleAtoms[1] != 0 || settleAtoms[2] != 1 || settleAtoms[3] != 2)
    {
        return formatString("molecule type %d must contain exactly one SETTLE over O/H1/H2", moleculeType);
    }

    const int parameterType = vsiteAtoms[0];
    if (parameterType < 0 || parameterType >= static_cast<int>(mtop.ffparams.iparams.size()))
    {
        return formatString("molecule type %d has an invalid VSITE3 parameter index", moleculeType);
    }
    const t_iparams& parameters = mtop.ffparams.iparams[parameterType];
    if (std::abs(parameters.vsite.a - c_opcWeight) > c_weightTolerance
        || std::abs(parameters.vsite.b - c_opcWeight) > c_weightTolerance)
    {
        return formatString("molecule type %d VSITE3 weights must both equal OPC value 0.1477224",
                            moleculeType);
    }

    if (moltype.atoms.atom == nullptr)
    {
        return formatString("molecule type %d has no atom properties", moleculeType);
    }
    if (moltype.atoms.atom[3].ptype != ParticleType::VSite || moltype.atoms.atom[3].m != 0
        || moltype.atoms.atom[0].ptype == ParticleType::VSite
        || moltype.atoms.atom[1].ptype == ParticleType::VSite
        || moltype.atoms.atom[2].ptype == ParticleType::VSite)
    {
        return formatString("molecule type %d requires a massless EPW virtual site and three real constructing atoms",
                            moleculeType);
    }
    if (moltype.atoms.atom[0].m <= 0 || moltype.atoms.atom[1].m <= 0 || moltype.atoms.atom[2].m <= 0)
    {
        return formatString("molecule type %d has non-positive O/H constructing-atom mass", moleculeType);
    }
    return {};
}

} // namespace

bool isContiguousOpcVsiteBlock(const ArrayRef<const OpcVsiteGpuData> sites)
{
    if (sites.empty())
    {
        return false;
    }
    const OpcVsiteGpuData& first = sites.front();
    for (std::size_t siteIndex = 0; siteIndex < sites.size(); ++siteIndex)
    {
        const OpcVsiteGpuData& site = sites[siteIndex];
        const int oxygen = first.oxygen + 4 * static_cast<int>(siteIndex);
        if (site.oxygen != oxygen || site.hydrogen1 != oxygen + 1 || site.hydrogen2 != oxygen + 2
            || site.virtualSite != oxygen + 3 || site.weightH1 != first.weightH1
            || site.weightH2 != first.weightH2)
        {
            return false;
        }
    }
    return true;
}

OpcVsiteAnalysis analyzeOpcGpuVsites(const gmx_mtop_t& mtop)
{
    OpcVsiteAnalysis result;

    if (mtop.intermolecular_ilist != nullptr && hasAnyVsiteInteraction(*mtop.intermolecular_ilist))
    {
        result.hasVirtualSites = true;
        result.rejectionReason = "intermolecular virtual sites are not supported by the OPC CUDA MVP";
        return result;
    }

    std::vector<bool> isOpcMoleculeType(mtop.moltype.size(), false);
    for (std::size_t moleculeType = 0; moleculeType < mtop.moltype.size(); ++moleculeType)
    {
        if (!hasAnyVsiteInteraction(mtop.moltype[moleculeType].ilist))
        {
            continue;
        }
        result.hasVirtualSites = true;
        if (const std::string reason = validateOpcMoleculeType(mtop, moleculeType); !reason.empty())
        {
            result.rejectionReason = reason;
            return result;
        }
        isOpcMoleculeType[moleculeType] = true;
    }

    int globalAtomOffset = 0;
    for (const gmx_molblock_t& moleculeBlock : mtop.molblock)
    {
        if (moleculeBlock.type < 0 || moleculeBlock.type >= static_cast<int>(mtop.moltype.size()))
        {
            result.rejectionReason = "topology contains an invalid molecule-block type";
            return result;
        }
        const int atomsPerMolecule = mtop.moltype[moleculeBlock.type].atoms.nr;
        if (isOpcMoleculeType[moleculeBlock.type])
        {
            const auto& vsiteAtoms =
                    mtop.moltype[moleculeBlock.type].ilist[InteractionFunction::VirtualSite3].iatoms;
            const auto& parameters = mtop.ffparams.iparams[vsiteAtoms[0]].vsite;
            for (int molecule = 0; molecule < moleculeBlock.nmol; ++molecule)
            {
                const int offset = globalAtomOffset + molecule * atomsPerMolecule;
                result.sites.push_back({ offset,
                                         offset + 1,
                                         offset + 2,
                                         offset + 3,
                                         static_cast<float>(parameters.a),
                                         static_cast<float>(parameters.b) });
            }
        }
        globalAtomOffset += atomsPerMolecule * moleculeBlock.nmol;
    }
    result.totalVirtualSiteCount = static_cast<int>(result.sites.size());

    if (result.hasVirtualSites && result.sites.empty())
    {
        result.rejectionReason = "virtual-site molecule type is not represented by a molecule block";
    }
    if (mtop.natoms != 0 && globalAtomOffset != mtop.natoms)
    {
        result.rejectionReason = "molecule blocks do not cover the global atom count";
    }
    return result;
}

} // namespace gmx
