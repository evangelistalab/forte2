#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "determinant/determinant.h"
#include "determinant/configuration.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m) {
    nb::class_<Determinant>(m, "Determinant")
        .def(nb::init<const Determinant&>())
        .def(
            "__init__",
            [](Determinant* d, std::string str) {
                new (d) Determinant{};
                d->clear();
                if (str.size() > Determinant::norb()) {
                    throw std::runtime_error("Determinant string must be of length " +
                                             std::to_string(Determinant::norb()));
                }
                for (int i = 0; i < str.size(); ++i) {
                    if (str[i] == '2') {
                        d->set_na(i, true);
                        d->set_nb(i, true);
                    } else if (str[i] == 'a') {
                        d->set_na(i, true);
                    } else if (str[i] == 'b') {
                        d->set_nb(i, true);
                    } else if (str[i] == '0') {
                        // do nothing, the orbital is empty
                    } else {
                        throw std::runtime_error(
                            "Determinant: Invalid character in determinant string: |" + str +
                            "> (all characters must be 0, 2, a, or b)");
                    }
                }
            },
            "str"_a, "Build a determinant from a string representation")
        .def_static("zero", &Determinant::zero, "Create a zero determinant with no electrons")
        .def_prop_ro_static(
            "maxnorb", [](nb::object /* self */) { return Determinant::norb(); },
            "The maximum number of orbitals supported by the Determinant class")
        .def(
            "__eq__", [](const Determinant& a, const Determinant& b) { return a == b; },
            "Check if two determinants are equal")
        .def(
            "__lt__", [](const Determinant& a, const Determinant& b) { return a < b; },
            "Check if a determinant is less than another determinant")
        .def(
            "__hash__", [](const Determinant& d) { return Determinant::Hash{}(d); },
            "Get the hash of the determinant")
        .def(
            "__repr__", [](Determinant& d) { return str(d); },
            "String representation of the determinant")
        .def(
            "set_na",
            [](Determinant& d, size_t n, bool value) {
                d.check_index_bounds(n, "(alpha orbital)");
                d.set_na(n, value);
            },
            "n"_a, "value"_a, "Set the occupation of an alpha orbital")
        .def(
            "set_nb",
            [](Determinant& d, size_t n, bool value) {
                d.check_index_bounds(n, "(beta orbital)");
                d.set_nb(n, value);
            },
            "n"_a, "value"_a, "Set the occupation of a beta orbital")
        .def(
            "na",
            [](const Determinant& d, size_t n) {
                d.check_index_bounds(n, "(alpha orbital)");
                return d.na(n);
            },
            "n"_a, "Is orbital n occupied by an alpha electron?")
        .def(
            "nb",
            [](const Determinant& d, size_t n) {
                d.check_index_bounds(n, "(beta orbital)");
                return d.nb(n);
            },
            "n"_a, "Is orbital n occupied by a beta electron?")
        .def("count_alpha", &Determinant::count_alpha, "Count the number of alpha electrons")
        .def("count_beta", &Determinant::count_beta, "Count the number of beta electrons")
        .def(
            "count", [](Determinant& d) { return d.count_alpha() + d.count_beta(); },
            "Count the total number of electrons")
        .def(
            "create_alpha",
            [](Determinant& d, size_t n) {
                d.check_index_bounds(n, "(alpha orbital)");
                return d.create_alpha(n);
            },
            "n"_a,
            "Apply an alpha creation operator to the determinant at the specified orbital index "
            "and return the sign")
        .def(
            "create_beta",
            [](Determinant& d, size_t n) {
                d.check_index_bounds(n, "(beta orbital)");
                return d.create_beta(n);
            },
            "n"_a,
            "Apply a beta creation operator to the determinant at the specified orbital index and "
            "return the sign")
        .def(
            "destroy_alpha",
            [](Determinant& d, size_t n) {
                d.check_index_bounds(n, "(alpha orbital)");
                return d.destroy_alpha(n);
            },
            "n"_a,
            "Apply an alpha destruction operator to the determinant at the specified orbital "
            "index and return the sign")
        .def(
            "destroy_beta",
            [](Determinant& d, size_t n) {
                d.check_index_bounds(n, "(beta orbital)");
                return d.destroy_beta(n);
            },
            "n"_a,
            "Apply a beta destruction operator to the determinant at the specified orbital index "
            "and return the sign")
        .def("spin_flip", &Determinant::spin_flip,
             "Spin flip the determinant, i.e., swap alpha and beta orbitals")
        .def(
            "slater_sign",
            [](const Determinant& d, size_t n) {
                d.check_index_bounds(n / 2, "(orbital index)");
                return d.slater_sign(n);
            },
            "Get the sign of the Slater determinant")
        .def(
            "slater_sign_aa",
            [](const Determinant& d, size_t n, size_t m) {
                d.check_index_bounds(n, "(first alpha orbital index)");
                d.check_index_bounds(m, "(second alpha orbital index)");
                return d.slater_sign_aa(n, m);
            },
            "n"_a, "m"_a, "Get the alpha-alpha pair sign of the Slater determinant")
        .def(
            "slater_sign_bb",
            [](const Determinant& d, size_t n, size_t m) {
                d.check_index_bounds(n, "(first beta orbital index)");
                d.check_index_bounds(m, "(second beta orbital index)");
                return d.slater_sign_bb(n, m);
            },
            "n"_a, "m"_a, "Get the beta-beta pair sign of the Slater determinant")
        .def(
            "slater_sign_reverse",
            [](const Determinant& d, size_t n) {
                d.check_index_bounds(n / 2, "(orbital index)");
                return d.slater_sign_reverse(n);
            },
            "Get the sign of the Slater determinant")
        .def(
            "str", [](const Determinant& d, size_t n) { return str(d, n); },
            "n"_a = Determinant::norb(), "Get the string representation of the Slater determinant");
}

void export_configuration_api(nb::module_& m) {
    nb::class_<Configuration>(m, "Configuration")
        .def(nb::init<>(), "Build an empty configuration")
        .def(nb::init<const Determinant&>(), "Build a configuration from a determinant")
        .def(
            "str", [](const Configuration& a, int n) { return str(a, n); },
            "n"_a = Configuration::norb(),
            "Get the string representation of the Slater determinant")
        .def("is_empty", &Configuration::is_empty, "n"_a, "Is orbital n empty?")
        .def("is_docc", &Configuration::is_docc, "n"_a, "Is orbital n doubly occupied?")
        .def("is_socc", &Configuration::is_socc, "n"_a, "Is orbital n singly occupied?")
        .def("set_occ", &Configuration::set_occ, "n"_a, "value"_a,
             "Set the occupation value of an orbital")
        .def("count_docc", &Configuration::count_docc,
             "Count the number of doubly occupied orbitals")
        .def("count_socc", &Configuration::count_socc,
             "Count the number of singly occupied orbitals")
        .def(
            "get_docc_vec",
            [](const Configuration& c) {
                int dim = c.count_docc();
                std::vector<int> l(dim);
                c.get_docc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the doubly occupied orbitals")
        .def(
            "get_socc_vec",
            [](const Configuration& c) {
                int dim = c.count_socc();
                std::vector<int> l(dim);
                c.get_socc_vec(Configuration::norb(), l);
                return l;
            },
            "Get a list of the singly occupied orbitals")
        .def(
            "__repr__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__str__", [](const Configuration& a) { return str(a); },
            "Get the string representation of the configuration")
        .def(
            "__eq__", [](const Configuration& a, const Configuration& b) { return a == b; },
            "Check if two configurations are equal")
        .def(
            "__lt__", [](const Configuration& a, const Configuration& b) { return a < b; },
            "Check if a configuration is less than another configuration")
        .def(
            "__hash__", [](const Configuration& a) { return Configuration::Hash()(a); },
            "Get the hash of the configuration");
}
} // namespace forte2
