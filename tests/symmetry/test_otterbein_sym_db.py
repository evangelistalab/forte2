"""
See readme.txt for details on the test set.
"""

import json
from importlib import resources
import pytest

from forte2 import System

with resources.files("forte2.data").joinpath("otterbein_symmetry_db.json").open(
    "r"
) as f:
    bfile = json.load(f)


def test_otterbein_mol_1():
    xyz = bfile["1"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["1"]["abelian_pg"].lower()


def test_otterbein_mol_2():
    xyz = bfile["2"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["2"]["abelian_pg"].lower()


def test_otterbein_mol_3():
    xyz = bfile["3"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["3"]["abelian_pg"].lower()


def test_otterbein_mol_4():
    xyz = bfile["4"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["4"]["abelian_pg"].lower()


def test_otterbein_mol_5():
    xyz = bfile["5"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["5"]["abelian_pg"].lower()


def test_otterbein_mol_6():
    xyz = bfile["6"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["6"]["abelian_pg"].lower()


def test_otterbein_mol_7():
    xyz = bfile["7"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["7"]["abelian_pg"].lower()


def test_otterbein_mol_8():
    xyz = bfile["8"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["8"]["abelian_pg"].lower()


def test_otterbein_mol_9():
    xyz = bfile["9"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["9"]["abelian_pg"].lower()


def test_otterbein_mol_10():
    xyz = bfile["10"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["10"]["abelian_pg"].lower()


def test_otterbein_mol_11():
    xyz = bfile["11"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["11"]["abelian_pg"].lower()


def test_otterbein_mol_12():
    xyz = bfile["12"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["12"]["abelian_pg"].lower()


def test_otterbein_mol_13():
    xyz = bfile["13"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["13"]["abelian_pg"].lower()


def test_otterbein_mol_14():
    xyz = bfile["14"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["14"]["abelian_pg"].lower()


def test_otterbein_mol_15():
    xyz = bfile["15"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["15"]["abelian_pg"].lower()


def test_otterbein_mol_16():
    xyz = bfile["16"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["16"]["abelian_pg"].lower()


def test_otterbein_mol_17():
    xyz = bfile["17"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["17"]["abelian_pg"].lower()


def test_otterbein_mol_18():
    xyz = bfile["18"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["18"]["abelian_pg"].lower()


def test_otterbein_mol_19():
    xyz = bfile["19"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["19"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_20():
    xyz = bfile["20"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["20"]["abelian_pg"].lower()


def test_otterbein_mol_21():
    xyz = bfile["21"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["21"]["abelian_pg"].lower()


def test_otterbein_mol_22():
    xyz = bfile["22"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["22"]["abelian_pg"].lower()


def test_otterbein_mol_23():
    xyz = bfile["23"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["23"]["abelian_pg"].lower()


def test_otterbein_mol_24():
    xyz = bfile["24"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["24"]["abelian_pg"].lower()


def test_otterbein_mol_25():
    xyz = bfile["25"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["25"]["abelian_pg"].lower()


def test_otterbein_mol_26():
    xyz = bfile["26"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["26"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_27():
    xyz = bfile["27"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["27"]["abelian_pg"].lower()


def test_otterbein_mol_28():
    xyz = bfile["28"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["28"]["abelian_pg"].lower()


def test_otterbein_mol_29():
    xyz = bfile["29"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["29"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_30():
    xyz = bfile["30"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["30"]["abelian_pg"].lower()


def test_otterbein_mol_31():
    xyz = bfile["31"]["xyz"]
    system = System(
        xyz=xyz,
        basis_set="sap_helfem_large",
        minao_basis_set=None,
        symmetry=True,
        symmetry_tol=1e-5,
    )
    assert system.point_group.lower() == bfile["31"]["abelian_pg"].lower()


def test_otterbein_mol_32():
    xyz = bfile["32"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["32"]["abelian_pg"].lower()


def test_otterbein_mol_33():
    xyz = bfile["33"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["33"]["abelian_pg"].lower()


def test_otterbein_mol_34():
    xyz = bfile["34"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["34"]["abelian_pg"].lower()


def test_otterbein_mol_35():
    xyz = bfile["35"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["35"]["abelian_pg"].lower()


def test_otterbein_mol_36():
    xyz = bfile["36"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["36"]["abelian_pg"].lower()


def test_otterbein_mol_37():
    xyz = bfile["37"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["37"]["abelian_pg"].lower()


def test_otterbein_mol_38():
    xyz = bfile["38"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["38"]["abelian_pg"].lower()


def test_otterbein_mol_39():
    xyz = bfile["39"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["39"]["abelian_pg"].lower()


def test_otterbein_mol_40():
    xyz = bfile["40"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["40"]["abelian_pg"].lower()


def test_otterbein_mol_41():
    xyz = bfile["41"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["41"]["abelian_pg"].lower()


def test_otterbein_mol_42():
    xyz = bfile["42"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["42"]["abelian_pg"].lower()


def test_otterbein_mol_43():
    xyz = bfile["43"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["43"]["abelian_pg"].lower()


def test_otterbein_mol_44():
    xyz = bfile["44"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["44"]["abelian_pg"].lower()


def test_otterbein_mol_45():
    xyz = bfile["45"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["45"]["abelian_pg"].lower()


def test_otterbein_mol_46():
    xyz = bfile["46"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["46"]["abelian_pg"].lower()


def test_otterbein_mol_47():
    xyz = bfile["47"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["47"]["abelian_pg"].lower()


def test_otterbein_mol_48():
    xyz = bfile["48"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["48"]["abelian_pg"].lower()


def test_otterbein_mol_49():
    xyz = bfile["49"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["49"]["abelian_pg"].lower()


def test_otterbein_mol_50():
    xyz = bfile["50"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["50"]["abelian_pg"].lower()


def test_otterbein_mol_51():
    xyz = bfile["51"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["51"]["abelian_pg"].lower()


def test_otterbein_mol_52():
    xyz = bfile["52"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["52"]["abelian_pg"].lower()


def test_otterbein_mol_53():
    xyz = bfile["53"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["53"]["abelian_pg"].lower()


def test_otterbein_mol_54():
    xyz = bfile["54"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["54"]["abelian_pg"].lower()


def test_otterbein_mol_55():
    xyz = bfile["55"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["55"]["abelian_pg"].lower()


def test_otterbein_mol_56():
    xyz = bfile["56"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["56"]["abelian_pg"].lower()


def test_otterbein_mol_57():
    xyz = bfile["57"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["57"]["abelian_pg"].lower()


def test_otterbein_mol_58():
    xyz = bfile["58"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["58"]["abelian_pg"].lower()


def test_otterbein_mol_59():
    xyz = bfile["59"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["59"]["abelian_pg"].lower()


def test_otterbein_mol_60():
    xyz = bfile["60"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["60"]["abelian_pg"].lower()


def test_otterbein_mol_61():
    xyz = bfile["61"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["61"]["abelian_pg"].lower()


def test_otterbein_mol_62():
    xyz = bfile["62"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["62"]["abelian_pg"].lower()


def test_otterbein_mol_63():
    xyz = bfile["63"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["63"]["abelian_pg"].lower()


def test_otterbein_mol_64():
    xyz = bfile["64"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["64"]["abelian_pg"].lower()


def test_otterbein_mol_65():
    xyz = bfile["65"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["65"]["abelian_pg"].lower()


def test_otterbein_mol_66():
    xyz = bfile["66"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["66"]["abelian_pg"].lower()


def test_otterbein_mol_67():
    xyz = bfile["67"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["67"]["abelian_pg"].lower()


def test_otterbein_mol_68():
    xyz = bfile["68"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["68"]["abelian_pg"].lower()


def test_otterbein_mol_69():
    xyz = bfile["69"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["69"]["abelian_pg"].lower()


def test_otterbein_mol_70():
    xyz = bfile["70"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["70"]["abelian_pg"].lower()


def test_otterbein_mol_71():
    xyz = bfile["71"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["71"]["abelian_pg"].lower()


def test_otterbein_mol_72():
    xyz = bfile["72"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["72"]["abelian_pg"].lower()


def test_otterbein_mol_73():
    xyz = bfile["73"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["73"]["abelian_pg"].lower()


def test_otterbein_mol_74():
    xyz = bfile["74"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["74"]["abelian_pg"].lower()


def test_otterbein_mol_75():
    xyz = bfile["75"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["75"]["abelian_pg"].lower()


def test_otterbein_mol_76():
    xyz = bfile["76"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["76"]["abelian_pg"].lower()


def test_otterbein_mol_77():
    xyz = bfile["77"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["77"]["abelian_pg"].lower()


def test_otterbein_mol_78():
    xyz = bfile["78"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["78"]["abelian_pg"].lower()


def test_otterbein_mol_79():
    xyz = bfile["79"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["79"]["abelian_pg"].lower()


def test_otterbein_mol_80():
    xyz = bfile["80"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["80"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_81():
    xyz = bfile["81"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["81"]["abelian_pg"].lower()


def test_otterbein_mol_82():
    xyz = bfile["82"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["82"]["abelian_pg"].lower()


def test_otterbein_mol_83():
    xyz = bfile["83"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["83"]["abelian_pg"].lower()


def test_otterbein_mol_84():
    xyz = bfile["84"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["84"]["abelian_pg"].lower()


def test_otterbein_mol_85():
    xyz = bfile["85"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["85"]["abelian_pg"].lower()


def test_otterbein_mol_86():
    xyz = bfile["86"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["86"]["abelian_pg"].lower()


def test_otterbein_mol_87():
    xyz = bfile["87"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["87"]["abelian_pg"].lower()


def test_otterbein_mol_88():
    xyz = bfile["88"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["88"]["abelian_pg"].lower()


def test_otterbein_mol_89():
    xyz = bfile["89"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["89"]["abelian_pg"].lower()


def test_otterbein_mol_90():
    xyz = bfile["90"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["90"]["abelian_pg"].lower()


def test_otterbein_mol_91():
    xyz = bfile["91"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["91"]["abelian_pg"].lower()


def test_otterbein_mol_92():
    xyz = bfile["92"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["92"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_93():
    xyz = bfile["93"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["93"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_94():
    xyz = bfile["94"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["94"]["abelian_pg"].lower()


def test_otterbein_mol_95():
    xyz = bfile["95"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["95"]["abelian_pg"].lower()


def test_otterbein_mol_96():
    xyz = bfile["96"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["96"]["abelian_pg"].lower()


def test_otterbein_mol_97():
    xyz = bfile["97"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["97"]["abelian_pg"].lower()


def test_otterbein_mol_98():
    xyz = bfile["98"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["98"]["abelian_pg"].lower()


def test_otterbein_mol_99():
    xyz = bfile["99"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["99"]["abelian_pg"].lower()


def test_otterbein_mol_100():
    xyz = bfile["100"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["100"]["abelian_pg"].lower()


def test_otterbein_mol_101():
    xyz = bfile["101"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["101"]["abelian_pg"].lower()


def test_otterbein_mol_102():
    xyz = bfile["102"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["102"]["abelian_pg"].lower()


def test_otterbein_mol_103():
    xyz = bfile["103"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["103"]["abelian_pg"].lower()


def test_otterbein_mol_104():
    xyz = bfile["104"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["104"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_105():
    xyz = bfile["105"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["105"]["abelian_pg"].lower()


def test_otterbein_mol_106():
    xyz = bfile["106"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["106"]["abelian_pg"].lower()


def test_otterbein_mol_107():
    xyz = bfile["107"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["107"]["abelian_pg"].lower()


def test_otterbein_mol_108():
    xyz = bfile["108"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["108"]["abelian_pg"].lower()


def test_otterbein_mol_109():
    xyz = bfile["109"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["109"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_110():
    xyz = bfile["110"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["110"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_111():
    xyz = bfile["111"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["111"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_112():
    xyz = bfile["112"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["112"]["abelian_pg"].lower()


def test_otterbein_mol_113():
    xyz = bfile["113"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["113"]["abelian_pg"].lower()


def test_otterbein_mol_114():
    xyz = bfile["114"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["114"]["abelian_pg"].lower()


def test_otterbein_mol_115():
    xyz = bfile["115"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["115"]["abelian_pg"].lower()


def test_otterbein_mol_116():
    xyz = bfile["116"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["116"]["abelian_pg"].lower()


def test_otterbein_mol_117():
    xyz = bfile["117"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["117"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_118():
    xyz = bfile["118"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["118"]["abelian_pg"].lower()


def test_otterbein_mol_119():
    xyz = bfile["119"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["119"]["abelian_pg"].lower()


def test_otterbein_mol_120():
    xyz = bfile["120"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["120"]["abelian_pg"].lower()


def test_otterbein_mol_121():
    xyz = bfile["121"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["121"]["abelian_pg"].lower()


def test_otterbein_mol_122():
    xyz = bfile["122"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["122"]["abelian_pg"].lower()


def test_otterbein_mol_123():
    xyz = bfile["123"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["123"]["abelian_pg"].lower()


def test_otterbein_mol_124():
    xyz = bfile["124"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["124"]["abelian_pg"].lower()


def test_otterbein_mol_125():
    xyz = bfile["125"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["125"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_126():
    xyz = bfile["126"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["126"]["abelian_pg"].lower()


def test_otterbein_mol_127():
    xyz = bfile["127"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["127"]["abelian_pg"].lower()


@pytest.mark.slow
def test_otterbein_mol_128():
    xyz = bfile["128"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["128"]["abelian_pg"].lower()


def test_otterbein_mol_129():
    xyz = bfile["129"]["xyz"]
    system = System(
        xyz=xyz, basis_set="sap_helfem_large", minao_basis_set=None, symmetry=True
    )
    assert system.point_group.lower() == bfile["129"]["abelian_pg"].lower()
