#!/bin/python

import basis_set_exchange as bse

BASIS_NAMES = [
    "sto-3g",
    "sto-6g",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "cc-pv5z",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
    "aug-cc-pv5z",
    "cc-pcvdz",
    "cc-pcvtz",
    "cc-pcvqz",
    "cc-pvtz-jkfit",
    "cc-pvqz-jkfit",
    "cc-pv5z-jkfit",
    "cc-pvtz-rifit",
    "cc-pvqz-rifit",
    "cc-pv5z-rifit",
    "sap_helfem_large",
]

def fetch_basis():
    for basis_name in BASIS_NAMES:
        try:
            optimize_general = bse.get_basis_family(basis_name).lower() == "dunning"
            if optimize_general:
                print(f"Optimizing general contraction for {basis_name}")
            basis = bse.get_basis(basis_name, fmt="json", optimize_general=optimize_general)
            with open(f"{basis_name}.json", "w") as f:
                f.write(basis)
        except Exception as e:
            print(f"Failed to fetch basis set {basis_name}: {e}")

if __name__ == "__main__":
    fetch_basis()