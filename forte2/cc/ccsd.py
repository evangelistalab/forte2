import numpy as np
from forte2.cc.cc import _SRCCBase

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
        self._ccs_transformation()
        # T1 residual
        r1 = self._t1_residual()
        # T2 residual
        r2 = self._t2_residual()
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
        t1, t2 = self.T

        chi_vv = self.ints['vv'] + np.einsum("anef,fn->ae", self.ints['vovv'], t1, optimize=True)
        chi_oo = self.ints['oo'] + np.einsum("mnif,fn->mi", self.ints['ooov'], t1, optimize=True)
        h_ov = self.ints['ov'] + np.einsum("mnef,fn->me", self.ints['oovv'], t1, optimize=True)
        h_oo = chi_oo + np.einsum("me,ei->mi", h_ov, t1, optimize=True)
        h_ooov = self.ints['ooov'] + np.einsum("mnfe,fi->mnie", self.ints['oovv'], t1, optimize=True)
        h_vovv = self.ints['vovv'] - np.einsum("mnfe,an->amef", self.ints['oovv'], t1, optimize=True)

        singles_res = -np.einsum("mi,am->ai", h_oo, t1, optimize=True)
        singles_res += np.einsum("ae,ei->ai", chi_vv, t1, optimize=True)
        singles_res += np.einsum("anif,fn->ai", self.ints['voov'], t1, optimize=True)
        singles_res += np.einsum("me,aeim->ai", h_ov, t2, optimize=True)
        singles_res -= 0.5 * np.einsum("mnif,afmn->ai", h_ooov, t2, optimize=True)
        singles_res += 0.5 * np.einsum("anef,efin->ai", h_vovv, t2, optimize=True)

        singles_res += self.ints['vo']

        return singles_res

    def _t2_residual(self):
        """Compute the projection of the CCSD Hamiltonian on doubles
            X[a, b, i, j] = < ijab | (H_N exp(T1+T2))_C | 0 >
        """
        X = self.intermediates
        t1, t2 = self.T
        # intermediates
        I_oo = X['oo'] + 0.5 * np.einsum("mnef,efin->mi", self.ints['oovv'], t2, optimize=True)
        I_vv = X['vv'] - 0.5 * np.einsum("mnef,afmn->ae", self.ints['oovv'], t2, optimize=True)
        I_voov = X['voov'] + 0.5 * np.einsum("mnef,afin->amie", self.ints['oovv'], t2, optimize=True)
        I_oooo = X['oooo'] + 0.5 * np.einsum("mnef,efij->mnij", self.ints['oovv'], t2, optimize=True)
        I_vooo = X['vooo'] + 0.5 * np.einsum('anef,efij->anij', self.ints['vovv'] + 0.5 * X['vovv'], t2, optimize=True)
        tau = 0.5 * t2 + np.einsum('ai,bj->abij', t1, t1, optimize=True)

        doubles_res = -0.5 * np.einsum("amij,bm->abij", I_vooo, t1, optimize=True)
        doubles_res += 0.5 * np.einsum("abie,ej->abij", X['vvov'], t1, optimize=True)
        doubles_res += 0.5 * np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
        doubles_res -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
        doubles_res += np.einsum("amie,ebmj->abij", I_voov, t2, optimize=True)
        doubles_res += 0.25 * np.einsum("abef,efij->abij", self.ints['vvvv'], tau, optimize=True)
        doubles_res += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)

        doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3)).conj()
        doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2)).conj()

        doubles_res += self.ints['vvoo']

        return doubles_res
    
    def _ccs_transformation(self):
        """
        Calculate the quantities related to the one-
        and two-body components of the CCS similarity-transformed 
        Hamiltonian, [H_N exp(T1)]_C, which serve as suitable 
        intermediates for constructing the CCSD amplitude equations.
            H1[:, :] ~ < p | [H_N exp(T1)]_C | q > (related to, not equal!)
            H2[:, :, :, :] ~ < pq | [H_N exp(T1)]_C | rs > (related to, not equal!)
        """
        X = self.intermediates
        t1, t2 = self.T

        # 1-body components
        temp = self.ints['ov'] + np.einsum("mnef,fn->me", self.ints['oovv'], t1, optimize=True)
        X['ov'] = temp

        temp = self.ints['vv'] + (
            np.einsum("anef,fn->ae", self.ints['vovv'], t1, optimize=True)
            - np.einsum("me,am->ae", X['ov'], t1, optimize=True)
        ) 
        X['vv'] = temp

        temp = self.ints['oo'] + (
            np.einsum("mnif,fn->mi", self.ints['ooov'], t1, optimize=True)
            + np.einsum("me,ei->mi", X['ov'], t1, optimize=True)
        ) 
        X['oo'] = temp

        # 2-body components
        temp = np.einsum("mnfe,fi->mnie", self.ints['oovv'], t1, optimize=True) 
        X['ooov'] = temp

        temp = (
                0.5 * self.ints['oooo'] 
                + np.einsum("nmje,ei->mnij", self.ints['ooov'] + 0.5 * X['ooov'], t1, optimize=True) # no(4)nu(1)
        )
        temp -= np.transpose(temp, (0, 1, 3, 2)).conj()
        X['oooo'] = temp

        temp = -np.einsum("mnfe,an->amef", self.ints['oovv'], t1, optimize=True) # no(2)nu(3)
        X['vovv'] = temp

        temp = self.ints['voov'] + (
                np.einsum("amfe,fi->amie", self.ints['vovv'] + 0.5 * X['vovv'], t1, optimize=True)
                - np.einsum("nmie,an->amie", self.ints['ooov'] + 0.5 * X['ooov'], t1, optimize=True)
        )
        X['voov'] = temp

        temp2 = self.ints['voov'] + 0.5 * np.einsum('amef,ei->amif', self.ints['vovv'], t1, optimize=True) # no(2)nu(3)
        temp3 = self.ints['oooo'] + np.einsum('mnie,ej->mnij', X['ooov'], t1, optimize=True) # no(4)nu(1)
        temp = 0.5 * self.ints['vooo'] + (
            np.einsum('amie,ej->amij', temp2, t1, optimize=True)
            -0.25 * np.einsum('mnij,am->anij', temp3, t1, optimize=True)
        ) 
        temp -= np.transpose(temp, (0, 1, 3, 2)).conj()
        X['vooo'] = temp

        temp2 = np.einsum('mnie,am->anie', self.ints['ooov'], t1, optimize=True)
        temp = self.ints['vvov'] + np.einsum("anie,bn->abie", temp2, t1, optimize=True) # no(1)nu(4)
        X['vvov'] = temp

        self.intermediates = X
