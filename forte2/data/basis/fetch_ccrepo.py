import ccrepo
import basis_set_exchange as bse

basis_sets = [
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
    "cc-pvdz-x2c",
    "cc-pvtz-x2c",
    "cc-pvqz-x2c",
    "cc-pv5z-x2c",
    "aug-cc-pvdz-x2c",
    "aug-cc-pvtz-x2c",
    "aug-cc-pvqz-x2c",
    "cc-pvdz-dk",
    "cc-pvtz-dk",
    "cc-pvqz-dk",
    "cc-pv5z-dk",
    "cc-pvtz_jkfit",
    "cc-pvqz_jkfit",
    "cc-pv5z_jkfit",
    "cc-pvtz_optri",
    "cc-pvqz_optri",
    "cc-pv5z_optri",
    "aug-cc-pvdz_optri",
    "aug-cc-pvtz_optri",
    "aug-cc-pvqz_optri",
    "aug-cc-pv5z_optri",
]

for basis in basis_sets:
    elements = ccrepo.get_elements(basis)
    bstr = ccrepo.fetch_basis(elements, 'cc-pVDZ')
    ccrepo.write_basis(bstr, f'{basis}.gbs', format='gaussian')
    bse.convert_formatted_basis_file(f'{basis}.gbs', f'{basis}.json', in_fmt='gaussian94', out_fmt='json')

