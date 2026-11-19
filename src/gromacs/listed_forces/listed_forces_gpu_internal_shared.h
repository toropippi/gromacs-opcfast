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
 *
 * \brief Implements generic GPU bonded functionality
 *
 * \author Andrey Alekseenko <al42and@gmail.com>
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_LISTED_FORCES_GPU_INTERNAL_SHARED_H
#define GMX_LISTED_FORCES_LISTED_FORCES_GPU_INTERNAL_SHARED_H

#include "config.h"

#include "gromacs/pbcutil/ishift.h"

#if GMX_GPU_CUDA
#    include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
#    include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#elif GMX_GPU_SYCL
#    include "gromacs/gpu_utils/sycl_kernel_utils.h"
#    include "gromacs/gpu_utils/vectype_ops_sycl.h"
#    include "gromacs/pbcutil/pbc_aiuc_sycl.h"
#elif GMX_GPU_HIP
#    include "gromacs/gpu_utils/hip_kernel_utils.h"
#    include "gromacs/gpu_utils/vectype_ops_hip.h"
#    include "gromacs/pbcutil/pbc_aiuc_hip.h"
#else
#    error Building GPU bonded for unsupported backend
#endif
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"

#include "listed_forces_gpu_impl.h"

#ifndef DOXYGEN

namespace gmx
{

static constexpr float c_deg2RadF = gmx::c_deg2Rad;
constexpr float        c_Pi       = M_PI;

/* Some SYCL targets have troubles optimizing the dynamic array
 * member access despite the fact that all the loops are unrolled.
 *
 * See https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
 * for a details on why dynamic access is problematic.
 *
 * This seems to affect:
 * - hipSYCL 0.9.4 + Clang 14-15 for AMD (but not NVIDIA),
 * - IntelLLVM 2023-02 for NVIDIA and Arc (but to a much lesser extent PVC).
 *
 * This wrapper avoid dynamic accesses into the array, replacing them
 * with a `switch` instead.
 *
 * Based on the optimization by AMD/StreamHPC for their HIP port.
 */
template<typename T>
struct FTypeArray
{
    static_assert(gmx::numFTypesOnGpu == 8 || gmx::numFTypesOnGpu == 10,
                  "Please update the member initializer list and the switch below");
    constexpr FTypeArray(const T in[gmx::numFTypesOnGpu]) :
        data{ in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
#if GMX_GPU_CUDA
              , in[8], in[9]
#endif
        }
    {
    }
    GMX_DEVICE_FUNC_ATTRIBUTE constexpr T operator[](int idx) const
    {
        switch (idx)
        {
            case 0: return data[0];
            case 1: return data[1];
            case 2: return data[2];
            case 3: return data[3];
            case 4: return data[4];
            case 5: return data[5];
            case 6: return data[6];
            case 7: return data[7];
#if GMX_GPU_CUDA
            case 8: return data[8];
            default: return data[9];
#else
            default: return data[7];
#endif
        }
    }
    T data[gmx::numFTypesOnGpu];
};

/* Harmonic */
template<bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void harmonic_gpu(const float             kA,
                                                          const float             xA,
                                                          const float             x,
                                                          DevicePrivatePtr<float> V,
                                                          DevicePrivatePtr<float> F)
{
    constexpr float half = 0.5F;
    float           dx   = x - xA;
    float           dx2  = dx * dx;

    *F = -kA * dx;
    if constexpr (calcEner)
    {
        *V = half * kA * dx2;
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void bonds_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    const int type = gm_forceatoms[3 * i];
    const int ai   = gm_forceatoms[3 * i + 1];
    const int aj   = gm_forceatoms[3 * i + 2];

    /* dx = xi - xj, corrected for periodic boundary conditions. */
    DeviceFloat3 dx;
    int          ki = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

    float dr2 = gmxDeviceNorm2(dx);
    float dr  = gmxDeviceSqrt(dr2);

    float vbond;
    float fbond;
    harmonic_gpu<calcEner>(
            gm_forceparams[type].harmonic.krA, gm_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

    if constexpr (calcEner)
    {
        *vtot_loc += vbond;
    }

    if (dr2 != 0.0F)
    {
        fbond *= gmxDeviceRSqrt(dr2);

        DeviceFloat3 fij = fbond * dx;
        staggeredAtomicAddForce(gm_f, fij, ai, localId);
        staggeredAtomicAddForce(gm_f, -fij, aj, localId);
        if constexpr (calcVir)
        {
            if (ki != gmx::c_centralShiftIndex)
            {
                atomicFetchAddLocal(sm_fShiftLoc, ki, fij);
                atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -fij);
            }
        }
    }
}

template<bool returnShift>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float bond_angle_gpu(const DeviceFloat4             xi,
                                                             const DeviceFloat4             xj,
                                                             const DeviceFloat4             xk,
                                                             const PbcAiuc&                 pbcAiuc,
                                                             DevicePrivatePtr<DeviceFloat3> r_ij,
                                                             DevicePrivatePtr<DeviceFloat3> r_kj,
                                                             DevicePrivatePtr<float>        costh,
                                                             DevicePrivatePtr<int>          t1,
                                                             DevicePrivatePtr<int>          t2)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh = gmxDeviceCosAngle(*r_ij, *r_kj);
    // Return value is the angle between the bonds i-j and j-k
    return gmxDeviceAcos(*costh);
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void angles_gpu(const int               i,
                                                        DevicePrivatePtr<float> vtot_loc,
                                                        const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                        const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                        const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                        DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                        DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                        const PbcAiuc&                pbcAiuc,
                                                        const int                     localId)
{
    DeviceInt4 angleData = loadInt4(gm_forceatoms, i);
    const int  type      = angleData[0];
    const int  ai        = angleData[1];
    const int  aj        = angleData[2];
    const int  ak        = angleData[3];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    float        cos_theta;
    int          t1;
    int          t2;
    float        theta = bond_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

    float va;
    float dVdt;
    harmonic_gpu<calcEner>(
            gm_forceparams[type].harmonic.krA, gm_forceparams[type].harmonic.rA * c_deg2RadF, theta, &va, &dVdt);

    if constexpr (calcEner)
    {
        *vtot_loc += va;
    }

    float cos_theta2 = cos_theta * cos_theta;
    if (cos_theta2 < 1.0F)
    {
        float st    = dVdt * gmxDeviceRSqrt(1.0F - cos_theta2);
        float sth   = st * cos_theta;
        float nrij2 = gmxDeviceNorm2(r_ij);
        float nrkj2 = gmxDeviceNorm2(r_kj);

        float nrij_1 = gmxDeviceRSqrt(nrij2);
        float nrkj_1 = gmxDeviceRSqrt(nrkj2);

        float cik = st * nrij_1 * nrkj_1;
        float cii = sth * nrij_1 * nrij_1;
        float ckk = sth * nrkj_1 * nrkj_1;

        DeviceFloat3 f_i = cii * r_ij - cik * r_kj;
        DeviceFloat3 f_k = ckk * r_kj - cik * r_ij;
        DeviceFloat3 f_j = -f_i - f_k;

        staggeredAtomicAddForce(gm_f, f_i, ai, localId);
        staggeredAtomicAddForce(gm_f, f_j, aj, localId);
        staggeredAtomicAddForce(gm_f, f_k, ak, localId);

        if constexpr (calcVir)
        {
            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
        }
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void
urey_bradley_gpu(const int                                 i,
                 DevicePrivatePtr<float>                   vtot_loc,
                 const DeviceGlobalPtr<const t_iatom>      gm_forceatoms,
                 const DeviceGlobalPtr<const t_iparams>    gm_forceparams,
                 const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                 DeviceGlobalPtr<DeviceFloat3>             gm_f,
                 DeviceLocalPtr<DeviceFloat3>              sm_fShiftLoc,
                 const PbcAiuc&                            pbcAiuc,
                 const int                                 localId)
{
    DeviceInt4 ubData = loadInt4(gm_forceatoms, i);
    const int  type   = ubData[0];
    const int  ai     = ubData[1];
    const int  aj     = ubData[2];
    const int  ak     = ubData[3];

    const float th0A = gm_forceparams[type].u_b.thetaA * c_deg2RadF;
    const float kthA = gm_forceparams[type].u_b.kthetaA;
    const float r13A = gm_forceparams[type].u_b.r13A;
    const float kUBA = gm_forceparams[type].u_b.kUBA;

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    float        cos_theta;
    int          t1;
    int          t2;
    float        theta = bond_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

    float va;
    float dVdt;
    harmonic_gpu<calcEner>(kthA, th0A, theta, &va, &dVdt);

    if (calcEner)
    {
        *vtot_loc += va;
    }

    DeviceFloat3 r_ik;
    int          ki = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

    float dr2 = gmxDeviceNorm2(r_ik);
    float dr  = dr2 * gmxDeviceRSqrt(dr2);

    float vbond;
    float fbond;
    harmonic_gpu<calcEner>(kUBA, r13A, dr, &vbond, &fbond);

    float cos_theta2 = cos_theta * cos_theta;

    DeviceFloat3 f_i = { 0.0F, 0.0F, 0.0F };
    DeviceFloat3 f_j = { 0.0F, 0.0F, 0.0F };
    DeviceFloat3 f_k = { 0.0F, 0.0F, 0.0F };

    if (cos_theta2 < 1.0F)
    {
        float st  = dVdt * gmxDeviceRSqrt(1.0F - cos_theta2);
        float sth = st * cos_theta;

        float nrkj2 = gmxDeviceNorm2(r_kj);
        float nrij2 = gmxDeviceNorm2(r_ij);

        float cik = st * gmxDeviceRSqrt(nrkj2 * nrij2);
        float cii = sth / nrij2;
        float ckk = sth / nrkj2;

        f_i = cii * r_ij - cik * r_kj;
        f_k = ckk * r_kj - cik * r_ij;
        f_j = -f_i - f_k;

        if constexpr (calcVir)
        {
            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, f_k);
        }
    }

    /* Time for the bond calculations */
    if (dr2 != 0.0F)
    {
        if constexpr (calcEner)
        {
            *vtot_loc += vbond;
        }

        fbond *= gmxDeviceRSqrt(dr2);
        DeviceFloat3 fik = fbond * r_ik;
        f_i += fik;
        f_k -= fik;


        if constexpr (calcVir)
        {
            if (ki != gmx::c_centralShiftIndex)
            {
                atomicFetchAddLocal(sm_fShiftLoc, ki, fik);
                atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -fik);
            }
        }
    }

    if ((cos_theta2 < 1.0F) || (dr2 != 0.0F))
    {
        staggeredAtomicAddForce(gm_f, f_i, ai, localId);
        staggeredAtomicAddForce(gm_f, f_k, ak, localId);
    }

    if (cos_theta2 < 1.0F)
    {
        staggeredAtomicAddForce(gm_f, f_j, aj, localId);
    }
}

template<bool returnShift, typename T>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float dih_angle_gpu(const T                        xi,
                                                            const T                        xj,
                                                            const T                        xk,
                                                            const T                        xl,
                                                            const PbcAiuc&                 pbcAiuc,
                                                            DevicePrivatePtr<DeviceFloat3> r_ij,
                                                            DevicePrivatePtr<DeviceFloat3> r_kj,
                                                            DevicePrivatePtr<DeviceFloat3> r_kl,
                                                            DevicePrivatePtr<DeviceFloat3> m,
                                                            DevicePrivatePtr<DeviceFloat3> n,
                                                            DevicePrivatePtr<int>          t1,
                                                            DevicePrivatePtr<int>          t2,
                                                            DevicePrivatePtr<int>          t3)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = gmxDeviceCrossProd(*r_ij, *r_kj);
    *n         = gmxDeviceCrossProd(*r_kj, *r_kl);
    float phi  = gmxDeviceAngle(*m, *n);
    float ipr  = gmxDeviceInternalProd(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;
    phi        = sign * phi;

    return phi;
}

template<bool returnShift, typename T>
GMX_DEVICE_FUNC_ATTRIBUTE static inline float dih_angle_gpu_sincos(const T        xi,
                                                                   const T        xj,
                                                                   const T        xk,
                                                                   const T        xl,
                                                                   const PbcAiuc& pbcAiuc,
                                                                   DevicePrivatePtr<DeviceFloat3> r_ij,
                                                                   DevicePrivatePtr<DeviceFloat3> r_kj,
                                                                   DevicePrivatePtr<DeviceFloat3> r_kl,
                                                                   DevicePrivatePtr<DeviceFloat3> m,
                                                                   DevicePrivatePtr<DeviceFloat3> n,
                                                                   DevicePrivatePtr<int>   t1,
                                                                   DevicePrivatePtr<int>   t2,
                                                                   DevicePrivatePtr<float> cosval)
{
    *t1 = pbcDxAiucGpu<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xj, *r_kj);
    pbcDxAiucGpu<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m = gmxDeviceCrossProd(*r_ij, *r_kj);
    *n = gmxDeviceCrossProd(*r_kj, *r_kl);

    DeviceFloat3 w = gmxDeviceCrossProd(*m, *n);

    float wlen = gmxDeviceSqrt(gmxDeviceNorm2(w));
    float s    = gmxDeviceInternalProd(*m, *n);

    float mLenSq = gmxDeviceNorm2(*m);
    float nLenSq = gmxDeviceNorm2(*n);
    float mnInv  = gmxDeviceRSqrt(mLenSq * nLenSq);

    *cosval      = s * mnInv;
    float sinval = wlen * mnInv;

    float ipr  = gmxDeviceInternalProd(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;

    return sign * sinval;
}

GMX_DEVICE_FUNC_ATTRIBUTE
static inline void dopdihs_gpu(const float             cpA,
                               const float             phiA,
                               const int               mult,
                               const float             phi,
                               DevicePrivatePtr<float> v,
                               DevicePrivatePtr<float> f)
{
    float mdphi = mult * phi - phiA * c_deg2RadF;
    float sdphi = gmxDeviceSin(mdphi);
    *v          = cpA * (1.0F + gmxDeviceCos(mdphi));
    *f          = -cpA * mult * sdphi;
}

template<bool calcVir>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void do_dih_fup_gpu(const int                     i,
                                                            const int                     j,
                                                            const int                     k,
                                                            const int                     l,
                                                            const float                   ddphi,
                                                            const DeviceFloat3            r_ij,
                                                            const DeviceFloat3            r_kj,
                                                            const DeviceFloat3            r_kl,
                                                            const DeviceFloat3            m,
                                                            const DeviceFloat3            n,
                                                            DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                            DeviceLocalPtr<DeviceFloat3> sm_fShiftLoc,
                                                            const PbcAiuc& pbcAiuc,
                                                            const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                            const int t1,
                                                            const int t2,
                                                            const int localId)
{
    float iprm  = gmxDeviceNorm2(m);
    float iprn  = gmxDeviceNorm2(n);
    float nrkj2 = gmxDeviceNorm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float        nrkj_1 = gmxDeviceRSqrt(nrkj2);
        float        nrkj_2 = nrkj_1 * nrkj_1;
        float        nrkj   = nrkj2 * nrkj_1;
        float        a      = -ddphi * nrkj / iprm;
        DeviceFloat3 f_i    = a * m;
        float        b      = ddphi * nrkj / iprn;
        DeviceFloat3 f_l    = b * n;
        float        p      = gmxDeviceInternalProd(r_ij, r_kj);
        p *= nrkj_2;
        float q = gmxDeviceInternalProd(r_kl, r_kj);
        q *= nrkj_2;
        DeviceFloat3 uvec = p * f_i;
        DeviceFloat3 vvec = q * f_l;
        DeviceFloat3 svec = uvec - vvec;
        DeviceFloat3 f_j  = f_i - svec;
        DeviceFloat3 f_k  = f_l + svec;

        staggeredAtomicAddForce(gm_f, f_i, i, localId);
        staggeredAtomicAddForce(gm_f, -f_j, j, localId);
        staggeredAtomicAddForce(gm_f, -f_k, k, localId);
        staggeredAtomicAddForce(gm_f, f_l, l, localId);

        if constexpr (calcVir)
        {
            DeviceFloat3 dx_jl;
            int          t3 = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            atomicFetchAddLocal(sm_fShiftLoc, t1, f_i);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -f_j);
            atomicFetchAddLocal(sm_fShiftLoc, t2, -f_k);
            atomicFetchAddLocal(sm_fShiftLoc, t3, f_l);
        }
    }
}

#if GMX_GPU_CUDA
/*! \brief Evaluate one CHARMM CMAP interaction on CUDA.
 *
 * The coefficient buffer contains 16 bicubic coefficients for every periodic grid cell,
 * precomputed on the host with the same transform used by cmap_dihs().
 */
template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void cmap_gpu(
        const int                                      interactionIndex,
        DevicePrivatePtr<float>                        vtotLoc,
        const DeviceGlobalPtr<const t_iatom>           forceAtoms,
        const DeviceGlobalPtr<const t_iparams>         forceParameters,
        const DeviceGlobalPtr<const float>             cmapCoefficients,
        const int                                      gridSpacing,
        const int                                      coefficientsPerMap,
        const DeviceGlobalPtr<const DeviceFloat4>      xq,
        DeviceGlobalPtr<DeviceFloat3>                  force,
        DeviceLocalPtr<DeviceFloat3>                   shiftForce,
        const PbcAiuc&                                 pbcAiuc,
        const int                                      localId)
{
    constexpr int c_stride = 6;
    const int     type     = forceAtoms[c_stride * interactionIndex];
    const int     ai       = forceAtoms[c_stride * interactionIndex + 1];
    const int     aj       = forceAtoms[c_stride * interactionIndex + 2];
    const int     ak       = forceAtoms[c_stride * interactionIndex + 3];
    const int     al       = forceAtoms[c_stride * interactionIndex + 4];
    const int     am       = forceAtoms[c_stride * interactionIndex + 5];

    DeviceFloat3 r1Ij, r1Kj, r1Kl, m1, n1;
    DeviceFloat3 r2Ij, r2Kj, r2Kl, m2, n2;
    int          t11, t21, t31, t12, t22, t32;

    const float phi1 = dih_angle_gpu<calcVir>(
            xq[ai], xq[aj], xq[ak], xq[al], pbcAiuc, &r1Ij, &r1Kj, &r1Kl, &m1, &n1, &t11, &t21, &t31);
    const float phi2 = dih_angle_gpu<calcVir>(
            xq[aj], xq[ak], xq[al], xq[am], pbcAiuc, &r2Ij, &r2Kj, &r2Kl, &m2, &n2, &t12, &t22, &t32);

    constexpr float twoPi = 2.0F * c_Pi;
    float xphi1 = phi1 + c_Pi;
    float xphi2 = phi2 + c_Pi;
    if (xphi1 < 0.0F)
    {
        xphi1 += twoPi;
    }
    else if (xphi1 >= twoPi)
    {
        xphi1 -= twoPi;
    }
    if (xphi2 < 0.0F)
    {
        xphi2 += twoPi;
    }
    else if (xphi2 >= twoPi)
    {
        xphi2 -= twoPi;
    }

    const float gridScale = static_cast<float>(gridSpacing) / twoPi;
    const float gridPhi1  = xphi1 * gridScale;
    const float gridPhi2  = xphi2 * gridScale;
    int         iphi1     = static_cast<int>(gridPhi1);
    int         iphi2     = static_cast<int>(gridPhi2);
    float       tt        = gridPhi1 - static_cast<float>(iphi1);
    float       tu        = gridPhi2 - static_cast<float>(iphi2);
    // Protect against a final-bit rounding to gridSpacing at the periodic boundary.
    if (iphi1 == gridSpacing)
    {
        iphi1 = 0;
        tt    = 0.0F;
    }
    if (iphi2 == gridSpacing)
    {
        iphi2 = 0;
        tu    = 0.0F;
    }

    const int cmapType = forceParameters[type].cmap.cmapA;
    const DeviceGlobalPtr<const float> coefficients =
            cmapCoefficients + cmapType * coefficientsPerMap
            + (iphi1 * gridSpacing + iphi2) * 16;

    float energy = 0.0F;
    float df1    = 0.0F;
    float df2    = 0.0F;
    for (int i = 3; i >= 0; i--)
    {
        const int row = 4 * i;
        energy = tt * energy
                 + ((coefficients[row + 3] * tu + coefficients[row + 2]) * tu
                    + coefficients[row + 1])
                           * tu
                 + coefficients[row];
        df1 = tu * df1
              + (3.0F * coefficients[12 + i] * tt + 2.0F * coefficients[8 + i]) * tt
              + coefficients[4 + i];
        df2 = tt * df2
              + (3.0F * coefficients[row + 3] * tu + 2.0F * coefficients[row + 2]) * tu
              + coefficients[row + 1];
    }
    df1 *= gridScale;
    df2 *= gridScale;

    if constexpr (calcEner)
    {
        *vtotLoc += energy;
    }

    do_dih_fup_gpu<calcVir>(ai,
                            aj,
                            ak,
                            al,
                            df1,
                            r1Ij,
                            r1Kj,
                            r1Kl,
                            m1,
                            n1,
                            force,
                            shiftForce,
                            pbcAiuc,
                            xq,
                            t11,
                            t21,
                            localId);
    do_dih_fup_gpu<calcVir>(aj,
                            ak,
                            al,
                            am,
                            df2,
                            r2Ij,
                            r2Kj,
                            r2Kl,
                            m2,
                            n2,
                            force,
                            shiftForce,
                            pbcAiuc,
                            xq,
                            t12,
                            t22,
                            localId);
}
#endif

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void pdihs_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[5 * i];
    int ai   = gm_forceatoms[5 * i + 1];
    int aj   = gm_forceatoms[5 * i + 2];
    int ak   = gm_forceatoms[5 * i + 3];
    int al   = gm_forceatoms[5 * i + 4];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    DeviceFloat3 r_kl;
    DeviceFloat3 m;
    DeviceFloat3 n;
    int          t1;
    int          t2;
    int          t3;
    float        phi = dih_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

    float vpd;
    float ddphi;
    dopdihs_gpu(gm_forceparams[type].pdihs.cpA,
                gm_forceparams[type].pdihs.phiA,
                gm_forceparams[type].pdihs.mult,
                phi,
                &vpd,
                &ddphi);

    if constexpr (calcEner)
    {
        *vtot_loc += vpd;
    }

    do_dih_fup_gpu<calcVir>(
            ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void rbdihs_gpu(const int               i,
                                                        DevicePrivatePtr<float> vtot_loc,
                                                        const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                        const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                        const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                        DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                        DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                        const PbcAiuc&                pbcAiuc,
                                                        const int                     localId)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;

    {
        int type = gm_forceatoms[5 * i];
        int ai   = gm_forceatoms[5 * i + 1];
        int aj   = gm_forceatoms[5 * i + 2];
        int ak   = gm_forceatoms[5 * i + 3];
        int al   = gm_forceatoms[5 * i + 4];

        DeviceFloat3 r_ij;
        DeviceFloat3 r_kj;
        DeviceFloat3 r_kl;
        DeviceFloat3 m;
        DeviceFloat3 n;
        int          t1;
        int          t2;

        // Changing the sign of sin and cos to convert to polymer convention
        float       cos_phi;
        const float negative_sin_phi = dih_angle_gpu_sincos<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &cos_phi);
        cos_phi *= -1;

        float parm[NR_RBDIHS];
#    pragma unroll NR_RBDIHS
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = gm_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if constexpr (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = ddphi * negative_sin_phi;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);
        if constexpr (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

//! Wrap angle from range [-3*pi; 3*pi) to [-pi; pi)
GMX_DEVICE_FUNC_ATTRIBUTE
static constexpr float wrapAngle(float a)
{
    constexpr float c_twoPi = 2.0F * c_Pi;
    if (a >= c_Pi)
    {
        return a - c_twoPi;
    }
    else if (a < -c_Pi)
    {
        return a + c_twoPi;
    }
    else
    {
        return a;
    }
}

template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void idihs_gpu(const int               i,
                                                       DevicePrivatePtr<float> vtot_loc,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_forceparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[5 * i];
    int ai   = gm_forceatoms[5 * i + 1];
    int aj   = gm_forceatoms[5 * i + 2];
    int ak   = gm_forceatoms[5 * i + 3];
    int al   = gm_forceatoms[5 * i + 4];

    DeviceFloat3 r_ij;
    DeviceFloat3 r_kj;
    DeviceFloat3 r_kl;
    DeviceFloat3 m;
    DeviceFloat3 n;
    int          t1;
    int          t2;
    int          t3;
    float        phi = dih_angle_gpu<calcVir>(
            gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

    /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
     * force changes if we just apply a normal harmonic.
     * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
     * This means we will never have the periodicity problem, unless
     * the dihedral is Pi away from phiO, which is very unlikely due to
     * the potential.
     */
    float kA = gm_forceparams[type].harmonic.krA;
    float pA = gm_forceparams[type].harmonic.rA;

    float phi0 = pA * c_deg2RadF;

    float dp = wrapAngle(phi - phi0);

    float ddphi = -kA * dp;

    do_dih_fup_gpu<calcVir>(
            ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, localId);

    if constexpr (calcEner)
    {
        *vtot_loc += -0.5F * ddphi * dp;
    }
}

template<bool calcVir, bool calcEner, bool useExplicitPairParameters = false>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void pairs_gpu(const int i,
                                                       const DeviceGlobalPtr<const t_iatom> gm_forceatoms,
                                                       const DeviceGlobalPtr<const t_iparams> gm_iparams,
                                                       const DeviceGlobalPtr<const DeviceFloat4> gm_xq,
                                                       DeviceGlobalPtr<DeviceFloat3> gm_f,
                                                       DeviceLocalPtr<DeviceFloat3>  sm_fShiftLoc,
                                                       const PbcAiuc&                pbcAiuc,
                                                       const float                   coulombConstant,
                                                       const float                   fudgeQQ,
                                                       DevicePrivatePtr<float>       vtot_loc,
                                                       DevicePrivatePtr<float>       vtotElec_loc,
                                                       const int                     localId)
{
    int type = gm_forceatoms[3 * i];
    int ai   = gm_forceatoms[3 * i + 1];
    int aj   = gm_forceatoms[3 * i + 2];

    float qq;
    float c6;
    float c12;
    float electrostaticsScaleFactor;
    if constexpr (useExplicitPairParameters)
    {
        qq                          = gm_iparams[type].ljc14.qi * gm_iparams[type].ljc14.qj;
        c6                          = gm_iparams[type].ljc14.c6;
        c12                         = gm_iparams[type].ljc14.c12;
        electrostaticsScaleFactor = coulombConstant * gm_iparams[type].ljc14.fqq;
    }
    else
    {
        qq                          = gm_xq[ai][3] * gm_xq[aj][3];
        c6                          = gm_iparams[type].lj14.c6A;
        c12                         = gm_iparams[type].lj14.c12A;
        electrostaticsScaleFactor = coulombConstant * fudgeQQ;
    }

    /* Do we need to apply full periodic boundary conditions? */
    DeviceFloat3 dr;
    int          fshift_index = pbcDxAiucGpu<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

    float r2    = gmxDeviceNorm2(dr);
    float rinv  = gmxDeviceRSqrt(r2);
    float rinv2 = rinv * rinv;
    float rinv6 = rinv2 * rinv2 * rinv2;

    /* Calculate the Coulomb force * r */
    float velec = electrostaticsScaleFactor * qq * rinv;

    /* Calculate the LJ force * r and add it to the Coulomb part */
    float fr = (12.0F * c12 * rinv6 - 6.0F * c6) * rinv6 + velec;

    float        finvr = fr * rinv2;
    DeviceFloat3 f     = finvr * dr;

    /* Add the forces */
    staggeredAtomicAddForce(gm_f, f, ai, localId);
    staggeredAtomicAddForce(gm_f, -f, aj, localId);
    if constexpr (calcVir)
    {
        if (fshift_index != gmx::c_centralShiftIndex)
        {
            atomicFetchAddLocal(sm_fShiftLoc, fshift_index, f);
            atomicFetchAddLocal(sm_fShiftLoc, gmx::c_centralShiftIndex, -f);
        }
    }

    // The elec and vdW contributions to the energy are separated only for the pairs
    // code, and combined later on.
    if constexpr (calcEner)
    {
        *vtot_loc += (c12 * rinv6 - c6) * rinv6;
        *vtotElec_loc += velec;
    }
}

/*! \brief Compatibility overload for GPU backends that currently support only standard LJ-14. */
template<bool calcVir, bool calcEner>
GMX_DEVICE_FUNC_ATTRIBUTE static inline void pairs_gpu(
        const int                                      i,
        const DeviceGlobalPtr<const t_iatom>           gm_forceatoms,
        const DeviceGlobalPtr<const t_iparams>         gm_iparams,
        const DeviceGlobalPtr<const DeviceFloat4>      gm_xq,
        DeviceGlobalPtr<DeviceFloat3>                  gm_f,
        DeviceLocalPtr<DeviceFloat3>                   sm_fShiftLoc,
        const PbcAiuc&                                 pbcAiuc,
        const float                                    electrostaticsScaleFactor,
        DevicePrivatePtr<float>                        vtot_loc,
        DevicePrivatePtr<float>                        vtotElec_loc,
        const int                                      localId)
{
    pairs_gpu<calcVir, calcEner, false>(i,
                                        gm_forceatoms,
                                        gm_iparams,
                                        gm_xq,
                                        gm_f,
                                        sm_fShiftLoc,
                                        pbcAiuc,
                                        electrostaticsScaleFactor,
                                        1.0F,
                                        vtot_loc,
                                        vtotElec_loc,
                                        localId);
}

} // namespace gmx

#endif

#endif
