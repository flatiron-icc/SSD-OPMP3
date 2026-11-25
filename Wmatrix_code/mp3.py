# Adapted from the CCSD code of PySCF 2.15.1 (update_amps of pyscf/cc/ccsd.py
# and uccsd.py; Copyright 2014-2021 The PySCF Developers), Apache License 2.0: see LICENSE-PySCF.

"""The MP amplitudes c^(1) and c^(2) of the paper (Eqs. 7 and 9), used by
pyscf_w_matrices.py to build the W matrices.

c^(2) can be obtained from one CCSD iteration with only the terms linear
in the amplitudes, starting from initial doubles amplitudes c^(1) and no singles.
This returns c^(1) + c^(2), and costs less than one CCSD iteration.

With density fitting, the integrals are those of PySCF's DF-CCSD.  Otherwise
only the blocks the update reads are built (oooo, ovov, oovv and ovvo of each
spin case), by PySCF's out-of-core transformation into a scratch HDF5 file,
and the <ab||cd> term is AO-direct (PySCF's direct CCSD) when its buffers fit
in the memory left of max_memory (at least PySCF's CC minimum, 2000 MB by
default), else from vvvv integrals on disk.  c^(1) and the <ab||cd> term
are computed by PySCF's CCSD code.  PySCF's Fock matrix is replaced by
diag(eps_p), with eps_p the semicanonical orbital energies.  Frozen orbitals
are left out throughout.
"""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo, lib
from pyscf.cc import ccsd, uccsd
from pyscf.lib import logger

# <ab||cd> term with conventional integrals: None = AO-direct when its buffers
# fit (make_eris), True / False force the AO-direct / stored-vvvv path
VVVV_DIRECT = None


def make_eris(mycc, mo_energy):
    """The integrals of the update (module docstring), with the Fock matrix
    diag(eps_p).  ``mo_energy`` holds the semicanonical orbital energies eps_p
    of all orbitals (one array, or an alpha and a beta array); frozen orbitals
    are dropped here.
    """
    mycc.e_hf = mycc._scf.e_tot                 # the reference energy, field included
    mask = mycc.get_frozen_mask()
    u = isinstance(mask, tuple)
    if getattr(mycc, "with_df", None) is not None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    else:
        mo = tuple(c[:, m] for c, m in zip(mycc.mo_coeff, mask)) if u else mycc.mo_coeff[:, mask]
        na, nb = mycc.nocc if u else (mycc.nocc,) * 2
        npairs = na * (na - 1) // 2 + nb * (nb - 1) // 2 + na * nb if u else na * (na + 1) // 2
        # memory left of max_memory, floored at PySCF's CC minimum as for its own buffers
        free = max(ccsd.MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
        need_mb = 4.0 * npairs * mycc.mol.nao_nr() ** 2 * 8 / 1e6        # AO-direct buffers
        mycc.direct = need_mb < free if VVVV_DIRECT is None else bool(VVVV_DIRECT)
        eris = (uccsd if u else ccsd)._ChemistsERIs()
        eris.mo_coeff, eris.nocc, eris.mol, eris.feri = mo, mycc.nocc, mycc.mol, lib.H5TmpFile()
        # (i p|q r) with i occupied: attribute -> spaces of p, q, r
        blk = {"oooo": "ooo", "oovv": "ovv", "ovov": "vov", "ovvo": "vvo"}
        if u:
            ma, mb = mo
            plan = [(ma, ma, na, na, blk), (mb, mb, nb, nb, {k.upper(): v for k, v in blk.items()}),
                    (ma, mb, na, nb, {"ooOO": "ooo", "ooVV": "ovv", "ovOV": "vov", "ovVO": "vvo"}),
                    (mb, ma, nb, na, {"OOvv": "ovv", "OVvo": "vvo"})]
            vvvv = [("vvvv", ma[:, na:], ma[:, na:]), ("VVVV", mb[:, nb:], mb[:, nb:]),
                    ("vvVV", ma[:, na:], mb[:, nb:])]
        else:
            plan, vvvv = [(mo, mo, na, na, blk)], [("vvvv", mo[:, na:], mo[:, na:])]
        sl = lambda x, n: slice(None, n) if x == "o" else slice(n, None)
        dim = lambda x, n, nocc: nocc if x == "o" else n - nocc
        for ml, mr, nl, nr, spec in plan:
            nml, nmr = ml.shape[1], mr.shape[1]
            out = {k: eris.feri.create_dataset(k, (nl, dim(P, nml, nl), dim(Q, nmr, nr), dim(R, nmr, nr)), "f8")
                   for k, (P, Q, R) in spec.items()}
            # left spin without virtuals (nl == nml, where ao2mo would also pack the (i p)
            # pairs): its blocks only meet empty amplitudes, so they stay zero
            if 0 < nl < nml:
                ao2mo.general(mycc.mol, (ml[:, :nl], ml, mr, mr), eris.feri, "rows")   # ao2mo's 2000 MB default
                pblk = max(1, int(0.1e6 * free / 8 / (1.5 * nmr ** 2)))             # p per read
                for i in range(nl):                        # occupied row i, in chunks of p
                    for p0, p1 in lib.prange(0, nml, pblk):
                        row = lib.unpack_tril(eris.feri["rows"][i * nml + p0:i * nml + p1]).reshape(-1, nmr, nmr)
                        for k, (P, Q, R) in spec.items():
                            ps = range(nml)[sl(P, nl)]
                            a, b = max(p0, ps.start), min(p1, ps.stop)
                            if a < b:
                                out[k][i, a - ps.start:b - ps.start] = row[a - p0:b - p0, sl(Q, nr), sl(R, nr)]
                del eris.feri["rows"]
            for k, d in out.items():
                setattr(eris, k, d)
        if not mycc.direct:                                # s4-packed, read by PySCF's _add_vvvv
            for k, cl, cr in vvvv:
                ao2mo.general(mycc.mol, (cl, cl, cr, cr), eris.feri, dataname=k)
                setattr(eris, k, eris.feri[k])
    if u:
        eris.mo_energy = tuple(np.asarray(e)[m] for e, m in zip(mo_energy, mask))
        eris.focka, eris.fockb = (np.diag(e) for e in eris.mo_energy)
        eris.fock = (eris.focka, eris.fockb)
    else:
        eris.mo_energy = np.asarray(mo_energy)[mask]
        eris.fock = np.diag(eris.mo_energy)
    return eris


def mp23_amplitudes(mycc, eris) -> tuple:
    """c^(1) and c^(2) (Eqs. 7 and 9).  Closed shell: one array c_ijab with
    i, a alpha and j, b beta.  Otherwise: the alpha-alpha, alpha-beta and
    beta-beta blocks."""
    c1 = mycc.init_amps(eris)[2]
    if isinstance(c1, tuple):
        c2 = update_amps_linear(mycc, c1, eris)
        for c2_block, c1_block in zip(c2, c1):
            c2_block -= c1_block
    else:
        c2 = update_amps_linear_r(mycc, c1, eris)
        c2 -= c1
    return c1, c2


# ---------------------------------------------------------------------------
# One CCSD iteration, linear terms only: for t = c^(1) it returns
# c^(1) + c^(2).  Variable names follow PySCF's update_amps.
# ---------------------------------------------------------------------------

def _antisymmetrize(u2: np.ndarray) -> np.ndarray:
    """Halve, then antisymmetrize in a, b and in i, j."""
    u2 *= .5
    u2 = u2 - u2.transpose(0, 1, 3, 2)
    return u2 - u2.transpose(1, 0, 2, 3)


def update_amps_linear(mycc, t2: tuple, eris) -> tuple:
    """Unrestricted version, from uccsd.update_amps."""
    t2aa, t2ab, t2bb = t2

    # 1/2 sum_cd <ab||cd> t_ijcd
    u2aa, u2ab, u2bb = mycc._add_vvvv(None, t2, eris)

    # same spin: 1/2 sum_kl <kl||ij> t_klab, and <ij||ab>
    for u2, t2ss, oooo, ovov in ((u2aa, t2aa, eris.oooo, eris.ovov),
                                 (u2bb, t2bb, eris.OOOO, eris.OVOV)):
        u2 *= .5
        u2 += lib.einsum("mnab,mnij->ijab", t2ss,
                         np.asarray(oooo).transpose(0, 2, 1, 3) * .5)
        ovov = np.asarray(ovov)
        u2 += (ovov - ovov.transpose(0, 3, 2, 1)).conj().transpose(0, 2, 1, 3) * .5
    ovov = None

    # opposite spin: sum_kl <kl|ij> t_klab, and <ij|ab>
    u2ab += lib.einsum("mNaB,mNiJ->iJaB", t2ab,
                       np.asarray(eris.ooOO).transpose(0, 2, 1, 3))
    u2ab += np.asarray(eris.ovOV).conj().transpose(0, 2, 1, 3)

    # sum_kc terms of Eq. 9
    wovvo = (np.asarray(eris.ovvo).transpose(0, 2, 1, 3)
             - np.asarray(eris.oovv).transpose(0, 2, 3, 1))
    wOVVO = (np.asarray(eris.OVVO).transpose(0, 2, 1, 3)
             - np.asarray(eris.OOVV).transpose(0, 2, 3, 1))
    woVvO = np.asarray(eris.ovVO).transpose(0, 2, 1, 3)
    woVVo = -np.asarray(eris.ooVV).transpose(0, 2, 3, 1)
    wOvVo = np.asarray(eris.OVvo).transpose(0, 2, 1, 3)
    wOvvO = -np.asarray(eris.OOvv).transpose(0, 2, 3, 1)

    u2aa += 2 * lib.einsum("imae,mbej->ijab", t2aa, wovvo)
    u2aa += 2 * lib.einsum("iMaE,MbEj->ijab", t2ab, wOvVo)
    u2bb += 2 * lib.einsum("imae,mbej->ijab", t2bb, wOVVO)
    u2bb += 2 * lib.einsum("mIeA,mBeJ->IJAB", t2ab, woVvO)
    u2ab += lib.einsum("imae,mBeJ->iJaB", t2aa, woVvO)
    u2ab += lib.einsum("iMaE,MBEJ->iJaB", t2ab, wOVVO)
    u2ab += lib.einsum("iMeA,MbeJ->iJbA", t2ab, wOvvO)
    u2ab += lib.einsum("IMAE,MbEj->jIbA", t2bb, wOvVo)
    u2ab += lib.einsum("mIeA,mbej->jIbA", t2ab, wovvo)
    u2ab += lib.einsum("mIaE,mBEj->jIaB", t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    # divide by eps_i + eps_j - eps_a - eps_b
    nocca, noccb = t2ab.shape[:2]
    eia_a, eia_b = (lib.direct_sum("i-a->ia", e[:no], e[no:])
                    for e, no in zip(eris.mo_energy, (nocca, noccb)))
    u2aa = _antisymmetrize(u2aa)
    u2bb = _antisymmetrize(u2bb)
    u2aa /= lib.direct_sum("ia+jb->ijab", eia_a, eia_a)
    u2ab /= lib.direct_sum("ia+jb->ijab", eia_a, eia_b)
    u2bb /= lib.direct_sum("ia+jb->ijab", eia_b, eia_b)
    return u2aa, u2ab, u2bb


def update_amps_linear_r(mycc, t2: np.ndarray, eris) -> np.ndarray:
    """Closed-shell version, from ccsd.update_amps, including its loop over
    blocks of virtual orbitals sized from max_memory."""
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t2.shape[1:3]

    # 1/2 sum_cd <ab||cd> t_ijcd
    t2new = mycc._add_vvvv(None, t2, eris, with_ovvv=False, t2sym="jiba")   # t1 = 0
    t2new *= .5  # halved: t2new + t2new.transpose(1, 0, 3, 2) at the end

    woooo = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()

    unit = nocc**2 * nvir * 7 + nocc**3 + nocc * nvir**2
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(ccsd.BLKMIN, int((max_memory * .9e6 / 8 - nocc**4) / unit)))
    log.debug1("max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
               max_memory, nocc, nvir, blksize)

    for p0, p1 in lib.prange(0, nvir, blksize):
        # <ij|ab>
        wVooV = -np.asarray(eris.oovv[:, :, p0:p1]).transpose(2, 0, 1, 3)
        eris_ovvo = np.asarray(eris.ovvo[:, p0:p1])
        t2new[:, :, p0:p1] += (eris_ovvo * .5).transpose(0, 3, 1, 2)
        wVOov = eris_ovvo.conj().transpose(1, 0, 3, 2) + wVooV * .5
        eris_ovvo = None

        # sum_kc terms of Eq. 9
        for q0, q1 in lib.prange(0, nvir, blksize):
            tmp = lib.einsum("jkca,ckib->jaib", t2[:, :, p0:p1, q0:q1], wVooV)
            t2new[:, :, q0:q1] += tmp.transpose(2, 0, 1, 3)
            tmp *= .5
            t2new[:, :, q0:q1] += tmp.transpose(0, 2, 1, 3)
            tmp = None
        wVooV = None
        for q0, q1 in lib.prange(0, nvir, blksize):
            theta = t2[:, :, p0:p1, q0:q1] * 2
            theta -= t2[:, :, p0:p1, q0:q1].transpose(1, 0, 2, 3)
            t2new[:, :, q0:q1] += lib.einsum("kica,ckjb->ijab", theta, wVOov)
            theta = None
        wVOov = None

    # sum_kl <kl|ij> t_klab
    for p0, p1 in lib.prange(0, nvir, blksize):
        t2new[:, :, p0:p1] += .5 * lib.einsum("ijkl,klab->ijab", woooo, t2[:, :, p0:p1])
    woooo = None

    # symmetrize in (i, a) <-> (j, b); divide by eps_i + eps_j - eps_a - eps_b
    eia = eris.mo_energy[:nocc, None] - eris.mo_energy[None, nocc:]
    for i in range(nocc):
        if i > 0:
            t2new[i, :i] += t2new[:i, i].transpose(0, 2, 1)
            t2new[i, :i] /= lib.direct_sum("a,jb->jab", eia[i], eia[:i])
            t2new[:i, i] = t2new[i, :i].transpose(0, 2, 1)
        t2new[i, i] = t2new[i, i] + t2new[i, i].T
        t2new[i, i] /= lib.direct_sum("a,b->ab", eia[i], eia[i])
    return t2new
