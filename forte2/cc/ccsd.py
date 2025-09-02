import time
import numpy as np
from forte2.cc.cc import _SRCCBase
from forte2.helpers import logger

class CCSD(_SRCCBase):

    def _build_energy_denominators(self):
        n = np.newaxis
        eps_o = np.diagonal(self.ints['oo'])
        eps_v = np.diagonal(self.ints['vv'])
        d1 = eps_o[n, :] - eps_v[:, n]
        d2 = eps_o[n, n, :, n] + eps_o[n, n, n, :] - eps_v[:, n, n, n] - eps_v[n, :, n, n]
        self.denom = (d1, d2)

    def _diis_update(self, diis):
        t1, t2 = self.T
        r1, r2 = self.r

        n1, n2 = np.prod(t1.shape), np.prod(t2.shape)

        T_flatten = np.hstack((t1.flatten(), t2.flatten()))
        R_flatten = np.hstack((r1.flatten(), r2.flatten()))
        T_extrap = diis.update(T_flatten, R_flatten)
        t1 = np.reshape(T_extrap[:n1], t1.shape)
        t2 = np.reshape(T_extrap[n1:], t2.shape)
        self.T = (t1, t2)

    def _build_residual(self):
        # CCS transformation
        tic = time.time()
        self._ccs_intermediates()
        logger.log_debug(f"[CCSD] time for CCS intermediates: {time.time() - tic}s")
        # T1 transformation of DF tensor
        tic = time.time()
        self._t1_transformation()
        logger.log_debug(f"[CCSD] time for DF T1 transform: {time.time() - tic}s")
        # T1 residual
        tic = time.time()
        r1 = self._t1_residual()
        logger.log_debug(f"[CCSD] time for T1 residual: {time.time() - tic}s")
        # T2 residual
        tic = time.time()
        r2 = self._t2_residual()
        logger.log_debug(f"[CCSD] time for T2 residual: {time.time() - tic}s")
        self.r = (r1, r2)

    def _build_update(self):
        t1, t2 = self.T
        r1, r2 = self.r
        d1, d2 = self.denom
        resnorm = 0.

        dt = r1 / (d1 - self.energy_shift)
        t1 += dt 
        resnorm += np.sum(dt**2)

        dt = r2 / (d2 - self.energy_shift)
        t2 += dt 
        resnorm += np.sum(dt**2)
        resnorm = np.sqrt(resnorm)
        self.T = (t1, t2)
        self.resnorm = resnorm

    def _build_energy(self):
        t1, t2 = self.T
        self.E = (
            np.einsum("me,em->", self.ints['ov'], t1, optimize=True)
            + 0.5 * np.einsum("mnef,em,fn->", self.ints['oovv'], t1, t1, optimize=True)
            + 0.25 * np.einsum("mnef,efmn->", self.ints['oovv'], t2, optimize=True)
        )

    def _build_initial_guess(self):
        _, d2 = self.denom
        t1 = np.zeros((self.nu, self.no))
        t2 = self.ints['vvoo'] / d2
        self.T = (t1, t2)

    def _t1_residual(self):
        """
        Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2))_C | 0 >
        """
        X = self.intermediates
        t1, t2 = self.T
        singles_residual = -np.einsum("mi,am->ai", X['oo'], t1, optimize=True)
        singles_residual += np.einsum("ae,ei->ai", X['vv'], t1, optimize=True)
        singles_residual += np.einsum("me,aeim->ai", X['ov'], t2, optimize=True) # [+]
        singles_residual += np.einsum("anif,fn->ai", self.ints['voov'], t1, optimize=True)
        singles_residual -= 0.5 * np.einsum("mnif,afmn->ai", self.ints['ooov'], t2, optimize=True)
        #
        b_vo = (
                  0.5 * np.einsum("xmf,efim->xei", self.ints.B['ov'], t2, optimize=True)
                - 0.5 * np.einsum("xme,efim->xfi", self.ints.B['ov'], t2, optimize=True)
        )
        singles_residual += np.einsum("xae,xei->ai", self.ints.B['vv'], b_vo, optimize=True)
        singles_residual += self.ints['vo']
        return singles_residual

    def _t2_residual(self):
        """Compute the projection of the CCSD Hamiltonian on doubles
            X[a, b, i, j] = < ijab | (H_N exp(T1+T2))_C | 0 >
        """
        t1, t2 = self.T
        X = self.intermediates
        # adjust (vv) intermediates
        X['vv'] -= np.einsum("me,am->ae", X['ov'], t1, optimize=True)
        # intermediates
        h_oooo = (
                np.einsum("xmi,xnj->mnij", self.BT1['oo'], self.BT1['oo'], optimize=True)
                - np.einsum("xmj,xni->mnij", self.BT1['oo'], self.BT1['oo'], optimize=True)
                + 0.5 * np.einsum("mnef,efij->mnij", self.ints['oovv'], t2, optimize=True)
        )
        h_voov = (
                np.einsum("xai,xme->amie", self.BT1['vo'], self.BT1['ov'], optimize=True)
                - np.einsum("xae,xmi->amie", self.BT1['vv'], self.BT1['oo'], optimize=True)
                + 0.5 * np.einsum("mnef,afin->amie", self.ints['oovv'], t2, optimize=True)
        )
        # <abij|H(1)|0>
        doubles_residual = 0.5 * np.einsum("xai,xbj->abij", self.BT1['vo'], self.BT1['vo'], optimize=True)
        # <abij|[H(1)*T2]_C|0>
        doubles_residual -= 0.5 * np.einsum("mi,abmj->abij", X['oo'], t2, optimize=True)
        doubles_residual += 0.5 * np.einsum("ae,ebij->abij", X['vv'], t2, optimize=True)
        doubles_residual += np.einsum("amie,ebmj->abij", h_voov, t2, optimize=True)
        doubles_residual += 0.125 * np.einsum("mnij,abmn->abij", h_oooo, t2, optimize=True)

        # [TODO]: Loop over slices of unoccupied orbitals and batch matrix multiply
        # vvvv term
        tic = time.time()
        self.BT1['vv'] = self.BT1['vv'].swapaxes(0, 1)
        # v(abef) t(efij) = [B(ae)B(bf) - B(af)B(be)] t(efij)
        for a in range(t1.shape[0]):
           for b in range(a + 1, t1.shape[0]):
               batch_ints = np.einsum("xe,xf->ef", self.BT1['vv'][a, :, :], self.BT1['vv'][b, :, :], optimize=True)
               batch_ints -= batch_ints.T.conj()
               doubles_residual[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, t2, optimize=True)
        # BT1['vv'] is not used again (it's recomputed later), so we don't actually have to swap it back
        # self.BT1['vv'] = self.BT1['vv'].swapaxes(0, 1) # pxq -> xpq
        logger.log_debug(f"[CCSD] time for vvvv contraction: {time.time() - tic} s")

        doubles_residual -= np.transpose(doubles_residual, (1, 0, 2, 3)).conj()
        doubles_residual -= np.transpose(doubles_residual, (0, 1, 3, 2)).conj()
        return doubles_residual
    
    def _ccs_intermediates(self):
        """
        Compute CCS-like intermediates.
        """
        X = self.intermediates
        t1, t2 = self.T

        # 1-body components
        temp = self.ints['ov'] + np.einsum("mnef,fn->me", self.ints['oovv'], t1, optimize=True)
        X['ov'] = temp

        temp = self.ints['vv'] - 0.5 * np.einsum("mnef,afmn->ae", self.ints['oovv'], t2, optimize=True)
        bt1 = np.einsum("xnf,fn->x", self.ints.B['ov'], t1, optimize=True)
        bxt1 = -np.einsum("xne,fn->xfe", self.ints.B['ov'], t1, optimize=True)
        temp += (
                 np.einsum("xae,x->ae", self.ints.B['vv'], bt1, optimize=True)
                + np.einsum("xaf,xfe->ae", self.ints.B['vv'], bxt1, optimize=True)
        )
        X['vv'] = temp

        temp = self.ints['oo'] + (
              np.einsum("mnif,fn->mi", self.ints['ooov'], t1, optimize=True)
            + np.einsum("me,ei->mi", X['ov'], t1, optimize=True)
            + 0.5 * np.einsum("mnef,efin->mi", self.ints['oovv'], t2, optimize=True)
        ) 
        X['oo'] = temp

        self.intermediates = X

    def _build_hbar(self):

        self.hbar = {}
        t1, t2 = self.T

        ### OV
        temp = self.ints['ov'] + np.einsum("imae,em->ia", self.ints['oovv'], t1, optimize=True)
        self.hbar['ov'] = temp
        ### OO
        temp = self.ints['oo'] + (
                np.einsum("je,ei->ji", self.hbar['ov'], t1, optimize=True)
                + np.einsum("jmie,em->ji", self.ints['ooov'], t1, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", self.ints['oovv'], t2, optimize=True)
        )
        self.hbar['oo'] = temp
        ### VV
        temp = self.ints['vv'] - 0.5 * np.einsum("mnef,afmn->ae", self.ints['oovv'], t2, optimize=True) 
        bt1 = np.einsum("xnf,fn->x", self.ints.B['ov'], t1, optimize=True)
        bxt1 = -np.einsum("xne,fn->xfe", self.ints.B['ov'], t1, optimize=True)
        temp += (
                np.einsum("xae,x->ae", self.ints.B['vv'], bt1, optimize=True)
                + np.einsum("xaf,xfe->ae", self.ints.B['vv'], bxt1, optimize=True)
                - np.einsum("me,am->ae", self.hbar['ov'], t1, optimize=True)
        )

        ### T1-transformation of Cholesky vectors
        # self._t1_transformation() # this is done at the end of obj.run() in cc.py
        # Cholesky-based intermediates folding some T2
        x_vo = self.BT1['vo'] + np.einsum("xnf,afin->xai", self.BT1['ov'], t2, optimize=True)

        ### TYPE: OOOO
        temp = (
                np.einsum("xmi,xnj->mnij", self.BT1['oo'], self.BT1['oo'], optimize=True)
                - np.einsum("xmj,xni->mnij", self.BT1['oo'], self.BT1['oo'], optimize=True)
                + 0.5 * np.einsum("mnef,efij->mnij", self.ints['oovv'], t2, optimize=True)
        )
        self.hbar['oooo'] = temp
        ### TYPE: OOOV
        temp = np.einsum("xmi,xne->mnie", self.BT1['oo'], self.BT1['ov'], optimize=True)
        temp -= np.transpose(temp, (1, 0, 2, 3))
        self.hbar['ooov'] = temp
        ### TYPE: VOVV
        temp = np.einsum("xbe,xnf->bnef", self.BT1['vv'], self.BT1['ov'], optimize=True)
        temp -= np.transpose(temp, (0, 1, 3, 2))
        self.hbar['vovv'] = temp
        ### TYPE: VOOO
        temp = (
                np.einsum("xai,xmj->amij", x_vo, self.BT1['oo'], optimize=True)
                + 0.25 * np.einsum("amef,efij->amij", self.hbar['vovv'], t2, optimize=True)
                + 0.5 * np.einsum("me,aeij->amij", self.hbar['ov'], t2, optimize=True)
                #
                # This exchange term won't go away!!!
                #
                - np.einsum("xnj,xmf,afin->amij", self.BT1['oo'], self.BT1['ov'], t2, optimize=True)
        )
        temp -= np.transpose(temp, (0, 1, 3, 2))
        self.hbar['vooo'] = temp
        ### TYPE: VOOV
        temp = (
                np.einsum("xai,xme->amie", self.BT1['vo'], self.BT1['ov'], optimize=True)
                - np.einsum("xae,xmi->amie", self.BT1['vv'], self.BT1['oo'], optimize=True)
                + np.einsum("mnef,afin->amie", self.ints['oovv'], t2, optimize=True)
        )
        self.hbar['voov'] = temp
        ### TYPE: VVOV
        temp = (
                np.einsum("xai,xbe->abie", x_vo, self.BT1['vv'], optimize=True)
                + 0.25 * np.einsum("mnie,abmn->abie", self.hbar['ooov'], t2, optimize=True)
                - 0.5 * np.einsum("me,abim->abie", self.hbar['ov'], t2, optimize=True)
                #
                # This exchange term won't go away! Cost is Naux*O^2*V^4, very expensive...
                #
                - np.einsum("xbf,xne,afin->abie", self.BT1['vv'], self.BT1['ov'], t2, optimize=True)
        )
        temp -= np.transpose(temp, (1, 0, 2, 3))
        self.hbar['vvov'] = temp
        ### TYPE: VVVV
        if self.build_hbar_nvirt >= 4:
            temp = (
                    np.einsum("xae,xbf->abef", self.BT1['vv'], self.BT1['vv'], optimize=True)
                    - np.einsum("xaf,xbe->abef", self.BT1['vv'], self.BT1['vv'], optimize=True)
                    + 0.5 * np.einsum("mnef,abmn->abef", self.ints['oovv'], t2, optimize=True)
            )
            self.hbar['vvvv'] = temp