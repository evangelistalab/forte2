import os
import numpy
import numpy.ctypeslib
import ctypes
import itertools
import sys
import scipy
import types
from .elements import Z_TO_ATOM_SYMBOL, ISOTOPE_MAIN


def load_library(libname):
    _loaderpath = os.path.dirname(__file__)
    return numpy.ctypeslib.load_library(libname, _loaderpath)


LIBCINT_AVAILABLE = False
try:
    load_library("libcgto")
    load_library("libcvhf")
    load_library("libnp_helper")
    LIBCINT_AVAILABLE = True
except Exception:
    pass


c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

PLAIN = 0
HERMITIAN = 1
ANTIHERMI = 2
SYMMETRIC = 3


# for _atm, _bas, _env
CHARGE_OF = 0
PTR_COORD = 1
NUC_MOD_OF = 2
PTR_ZETA = 3
PTR_FRAC_CHARGE = 4
PTR_RADIUS = 5
ATM_SLOTS = 6
ATOM_OF = 0
ANG_OF = 1
NPRIM_OF = 2
NCTR_OF = 3
RADI_POWER = 3  # for ECP
KAPPA_OF = 4
SO_TYPE_OF = 4  # for ECP
PTR_EXP = 5
PTR_COEFF = 6
BAS_SLOTS = 8
# pointer to env
PTR_EXPCUTOFF = 0
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG = 4
PTR_RINV_ZETA = 7
PTR_RANGE_OMEGA = 8
PTR_F12_ZETA = 9
PTR_GTG_ZETA = 10
NGRIDS = 11
PTR_GRIDS = 12
AS_RINV_ORIG_ATOM = 17
AS_ECPBAS_OFFSET = 18
AS_NECPBAS = 19
PTR_ENV_START = 20

# parameters from libcint
NUC_POINT = 1
NUC_GAUSS = 2
# nucleus with fractional charges. It can be used to mimic MM particles
NUC_FRAC_CHARGE = 3
NUC_ECP = 4  # atoms with pseudo potential

NORMALIZE_GTO = True


def len_spinor(l, kappa):
    """The number of spinor associated with given angular momentum and kappa.  If kappa is 0,
    return 4l+2
    """
    if kappa == 0:
        n = l * 4 + 2
    elif kappa < 0:
        n = l * 2 + 2
    else:
        n = l * 2
    return n


def flatten(lst):
    """flatten nested lists
    x[0] + x[1] + x[2] + ...

    Examples:

    >>> flatten([[0, 2], [1], [[9, 8, 7]]])
    [0, 2, 1, [9, 8, 7]]
    """
    return list(itertools.chain.from_iterable(lst))


def atom_basis_to_bas(atom_basis):
    bas = {}
    for atom_id, basis in atom_basis.items():
        atom_symbol = Z_TO_ATOM_SYMBOL[atom_id].capitalize()
        for shell in basis:
            angular_momentum = list(map(int, shell["angular_momentum"]))
            exponents = list(map(float, shell["exponents"]))
            if atom_symbol not in bas:
                bas[atom_symbol] = []
            for l, subshell_coefficients in itertools.zip_longest(
                angular_momentum,
                shell["coefficients"],
                fillvalue=angular_momentum[-1],
            ):
                coefficients = list(map(float, subshell_coefficients))
                bas[atom_symbol].append([l])
                for exp, coeff in zip(exponents, coefficients):
                    if abs(coeff) < 1e-10:
                        continue
                    bas[atom_symbol][-1].append([exp, coeff])

    return bas


def _rm_digit(symb):
    if symb.isalpha():
        return symb
    else:
        return "".join([i for i in symb if i.isalpha()])


def make_env(atoms, basis, pre_env=[], nucmod={}):
    """Generate the input arguments for ``libcint`` library based on internal
    format :attr:`Mole._atom` and :attr:`Mole._basis`
    """
    _atm = []
    _bas = []
    _env = [pre_env]
    ptr_env = len(pre_env)

    for ia, atom in enumerate(atoms):
        prop = {}
        symb = Z_TO_ATOM_SYMBOL[atom[0]].capitalize()
        nuclear_model = NUC_POINT
        if nucmod:
            if nucmod is None:
                nuclear_model = NUC_POINT
            elif isinstance(nucmod, (int, str, types.FunctionType)):
                nuclear_model = _parse_nuc_mod(nucmod)
            elif ia + 1 in nucmod:
                nuclear_model = _parse_nuc_mod(nucmod[ia + 1])
            elif symb in nucmod:
                nuclear_model = _parse_nuc_mod(nucmod[symb])
        atm0, env0 = make_atm_env(atom, ptr_env, nuclear_model, prop)
        ptr_env = ptr_env + len(env0)
        _atm.append(atm0)
        _env.append(env0)

    _basdic = {}
    for symb, basis_add in basis.items():
        bas0, env0 = make_bas_env(basis_add, 0, ptr_env)
        ptr_env = ptr_env + len(env0)
        _basdic[symb] = bas0
        _env.append(env0)

    for ia, atom in enumerate(atoms):
        symb = Z_TO_ATOM_SYMBOL[atom[0]].capitalize()
        puresymb = _rm_digit(symb)
        if symb in _basdic:
            b = _basdic[symb].copy()
        elif puresymb in _basdic:
            b = _basdic[puresymb].copy()
        else:
            if symb[:2].upper() == "X-":
                symb = symb[2:]
            elif symb[:6].upper() == "GHOST-":
                symb = symb[6:]
            puresymb = _rm_digit(symb)
            if symb in _basdic:
                b = _basdic[symb].copy()
            elif puresymb in _basdic:
                b = _basdic[puresymb].copy()
            else:
                sys.stderr.write(
                    "Warning: Basis not found for atom %d %s\n" % (ia, symb)
                )
                continue
        b[:, ATOM_OF] = ia
        _bas.append(b)

    if _atm:
        _atm = numpy.asarray(numpy.vstack(_atm), numpy.int32).reshape(-1, ATM_SLOTS)
    else:
        _atm = numpy.zeros((0, ATM_SLOTS), numpy.int32)
    if _bas:
        _bas = numpy.asarray(numpy.vstack(_bas), numpy.int32).reshape(-1, BAS_SLOTS)
    else:
        _bas = numpy.zeros((0, BAS_SLOTS), numpy.int32)
    _env = numpy.asarray(numpy.hstack(_env), dtype=numpy.float64)
    return _atm, _bas, _env


# append (charge, pointer to coordinates, nuc_mod) to _atm
def make_atm_env(atom, ptr=0, nuclear_model=NUC_POINT, nucprop={}):
    """Convert the internal format :attr:`Mole._atom` to the format required
    by ``libcint`` integrals
    """
    nuc_charge = atom[0]
    if nuclear_model == NUC_POINT:
        zeta = 0
    elif nuclear_model == NUC_GAUSS:
        zeta = dyall_nuc_mod(nuc_charge, nucprop)
    else:  # callable(nuclear_model)
        zeta = nuclear_model(nuc_charge, nucprop)
        nuclear_model = NUC_GAUSS
    _env = numpy.hstack((atom[1], zeta))
    _atm = numpy.zeros(6, dtype=numpy.int32)
    _atm[CHARGE_OF] = nuc_charge
    _atm[PTR_COORD] = ptr
    _atm[NUC_MOD_OF] = nuclear_model
    _atm[PTR_ZETA] = ptr + 3
    return _atm, _env


# append (atom, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0) to bas
# absorb normalization into GTO contraction coefficients
def make_bas_env(basis_add, atom_id=0, ptr=0):
    """Convert :attr:`Mole.basis` to the argument ``bas`` for ``libcint`` integrals"""
    _bas = []
    _env = []
    for b in basis_add:
        angl = b[0]
        if angl > 14:
            sys.stderr.write(
                "Warning: integral library does not support basis "
                "with angular momentum > 14\n"
            )

        if isinstance(b[1], (int, numpy.integer)):
            kappa = b[1]
            b_coeff = numpy.array(sorted(b[2:], reverse=True))
        else:
            kappa = 0
            b_coeff = numpy.array(sorted(b[1:], reverse=True))
        es = b_coeff[:, 0]
        cs = b_coeff[:, 1:]
        nprim, nctr = cs.shape
        cs = numpy.einsum("pi,p->pi", cs, gto_norm(angl, es))
        if NORMALIZE_GTO:
            cs = _nomalize_contracted_ao(angl, es, cs)

        _env.append(es)
        _env.append(cs.T.reshape(-1))
        ptr_exp = ptr
        ptr_coeff = ptr_exp + nprim
        ptr = ptr_coeff + nprim * nctr
        _bas.append([atom_id, angl, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])
    _env = flatten(_env)  # flatten nested lists
    return (
        numpy.array(_bas, numpy.int32).reshape(-1, BAS_SLOTS),
        numpy.array(_env, numpy.double),
    )


def _nomalize_contracted_ao(l, es, cs):
    # ee = numpy.empty((nprim,nprim))
    # for i in range(nprim):
    #    for j in range(i+1):
    #        ee[i,j] = ee[j,i] = gaussian_int(angl*2+2, es[i]+es[j])
    # s1 = 1/numpy.sqrt(numpy.einsum('pi,pq,qi->i', cs, ee, cs))
    ee = es.reshape(-1, 1) + es.reshape(1, -1)
    ee = gaussian_int(l * 2 + 2, ee)
    s1 = 1.0 / numpy.sqrt(numpy.einsum("pi,pq,qi->i", cs, ee, cs))
    return numpy.einsum("pi,i->pi", cs, s1)


def gaussian_int(n, alpha):
    r"""int_0^inf x^n exp(-alpha x^2) dx"""
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2.0 * alpha**n1)


def dyall_nuc_mod(nuc_charge, nucprop={}):
    """Generate the nuclear charge distribution parameter zeta
    rho(r) = nuc_charge * Norm * exp(-zeta * r^2)

    Ref. L. Visscher and K. Dyall, At. Data Nucl. Data Tables, 67, 207 (1997)
    """
    mass = nucprop.get("mass", ISOTOPE_MAIN[nuc_charge])
    r = (0.836 * mass ** (1.0 / 3) + 0.570) / 52917.7249
    zeta = 1.5 / (r**2)
    return zeta


def gto_norm(l, expnt):
    r"""Normalized factor for GTO radial part   :math:`g=r^l e^{-\alpha r^2}`

    .. math::

        \frac{1}{\sqrt{\int g^2 r^2 dr}}
        = \sqrt{\frac{2^{2l+3} (l+1)! (2a)^{l+1.5}}{(2l+2)!\sqrt{\pi}}}

    Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1995), 83-87.

    Args:
        l (int):
            angular momentum
        expnt :
            exponent :math:`\alpha`

    Returns:
        normalization factor

    Examples:

    >>> print(gto_norm(0, 1))
    2.5264751109842591
    """
    if numpy.all(l >= 0):
        # f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
        #        / (math.factorial(2*l+2) * math.sqrt(math.pi))
        # return math.sqrt(f)
        return 1 / numpy.sqrt(gaussian_int(l * 2 + 2, 2 * expnt))
    else:
        raise ValueError("l should be >= 0")


def _parse_nuc_mod(str_or_int_or_fn):
    nucmod = NUC_POINT
    if callable(str_or_int_or_fn):
        nucmod = str_or_int_or_fn
    elif (
        isinstance(str_or_int_or_fn, str) and str_or_int_or_fn[0].upper() == "G"
    ):  # 'gauss_nuc'
        nucmod = NUC_GAUSS
    elif str_or_int_or_fn != 0:
        nucmod = NUC_GAUSS
    return nucmod
