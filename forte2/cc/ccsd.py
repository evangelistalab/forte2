import time
import numpy as np
from forte2.cc.cc import _SRCCBase
from forte2.helpers import logger

class CCSD(_SRCCBase):

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
        logger.log_debug(f"time for CCS intermediates: {time.time() - tic}s")
        # T1 transformation of DF tensor
        tic = time.time()
        self._t1_transformation()
        logger.log_debug(f"time for DF T1 transform: {time.time() - tic}s")
        # T1 residual
        tic = time.time()
        r1 = self._t1_residual()
        logger.log_debug(f"time for T1 residual: {time.time() - tic}s")
        # T2 residual
        tic = time.time()
        r2 = self._t2_residual()
        logger.log_debug(f"time for T2 residual: {time.time() - tic}s")
        self.r = (r1, r2)

    def _build_update(self):
        t1, t2 = self.T
        r1, r2 = self.r
        resnorm = 0.
        for a in range(self.nu):
            for i in range(self.no):
                denom = self.ints['oo'][i, i] - self.ints['vv'][a, a]
                dt = r1[a, i] / (denom - self.energy_shift)
                t1[a, i] += dt
                resnorm += np.abs(dt)
        for a in range(self.nu):
            for b in range(a + 1, self.nu):
                for i in range(self.no):
                    for j in range(i + 1, self.no):
                        denom = self.ints['oo'][i, i] + self.ints['oo'][j, j] - self.ints['vv'][a, a] - self.ints['vv'][b, b]
                        dt = r2[a, b, i, j] / (denom - self.energy_shift)
                        t2[a, b, i, j] +=  dt
                        t2[b, a, i, j] = -t2[a, b, i, j]
                        t2[a, b, j, i] = -t2[a, b, i, j]
                        t2[b, a, j, i] = t2[a, b, i, j]
                        resnorm += np.abs(dt)
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
        t1 = np.zeros((self.nu, self.no))
        t2 = np.zeros((self.nu, self.nu, self.no, self.no))
        for a in range(self.nu):
            for b in range(a + 1, self.nu):
                for i in range(self.no):
                    for j in range(i + 1, self.no):
                        denom = self.ints['oo'][i, i] + self.ints['oo'][j, j] - self.ints['vv'][a, a] - self.ints['vv'][b, b]
                        t2[a, b, i, j] = self.ints['vvoo'][a, b, i, j] / (denom)

                        t2[b, a, i, j] = -t2[a, b, i, j]
                        t2[a, b, j, i] = -t2[a, b, i, j]
                        t2[b, a, j, i] = t2[a, b, i, j]
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
        # vvvv term
        for a in range(t1.shape[0]):
           for b in range(a + 1, t1.shape[0]):
               # <ab|ef> = <x|ae><x|bf>
               batch_ints = np.einsum("xe,xf->ef", self.BT1['vv'][:, a, :], self.BT1['vv'][:, b, :], optimize=True)
               batch_ints -= batch_ints.T.conj()
               doubles_residual[a, b, :, :] += 0.25 * np.einsum("ef,efij->ij", batch_ints, t2, optimize=True)
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

    def _t1_transformation(self):
        t1, _ = self.T
        # T1-transform Cholesky vectors
        self.BT1['ov'] = self.ints.B['ov'].copy()
        self.BT1['oo'] = self.ints.B['oo'].copy() + np.einsum("xme,ei->xmi", self.BT1['ov'], t1, optimize=True)
        self.BT1['vv'] = self.ints.B['vv'].copy() - np.einsum("xme,am->xae", self.BT1['ov'], t1, optimize=True)
        self.BT1['vo'] = (
                self.ints.B['vo'].copy()
                - np.einsum("xmi,am->xai", self.BT1['oo'], t1, optimize=True)
                + np.einsum("xae,ei->xai", self.BT1['vv'], t1, optimize=True)
                + np.einsum("xme,ei,am->xai", self.BT1['ov'], t1, t1, optimize=True)
        )
