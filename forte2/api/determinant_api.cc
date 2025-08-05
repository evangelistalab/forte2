#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ci/determinant.h"
#include "ci/configuration.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m) {
    nb::class_<Determinant>(m, "Determinant")
        .def(nb::init<const Determinant&>())
        .def("__init__",
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
                     }
                 }
             })
        // define a static method to create a zero determinant
        .def_static("zero", &Determinant::zero)
        .def("__eq__", [](const Determinant& a, const Determinant& b) { return a == b; })
        .def("__lt__", [](const Determinant& a, const Determinant& b) { return a < b; })
        .def("__hash__",
             [](const Determinant& d) {
                 // Use the hash function defined in the Determinant class
                 return Determinant::Hash{}(d);
             })
        .def(
            "__repr__", [](Determinant& d) { return str(d); },
            "String representation of the determinant")
        .def("set_na", &Determinant::set_na)
        .def("set_nb", &Determinant::set_nb)
        .def("na", &Determinant::na)
        .def("nb", &Determinant::nb)
        .def("count_a", &Determinant::count_a)
        .def("count_b", &Determinant::count_b)
        .def("count", [](Determinant& d) { return d.count_a() + d.count_b(); })
        .def("create_a", &Determinant::create_a, "n"_a,
             "Apply an alpha creation operator to the determinant at the specified orbital index "
             "and return the sign")
        .def("create_b", &Determinant::create_b, "n"_a,
             "Apply a beta creation operator to the determinant at the specified orbital index and "
             "return the sign")
        .def("destroy_a", &Determinant::destroy_a, "n"_a,
             "Apply an alpha destruction operator to the determinant at the specified orbital "
             "index and return the sign")
        .def("destroy_b", &Determinant::destroy_b, "n"_a,
             "Apply a beta destruction operator to the determinant at the specified orbital index "
             "and return the sign")
        .def("spin_flip", &Determinant::spin_flip,
             "Spin flip the determinant, i.e., swap alpha and beta orbitals")
        .def(
            "slater_sign", [](const Determinant& d, size_t n) { return d.slater_sign(n); },
            "Get the sign of the Slater determinant")
        .def(
            "slater_sign_reverse",
            [](const Determinant& d, size_t n) { return d.slater_sign_reverse(n); },
            "Get the sign of the Slater determinant")
        .def(
            "gen_excitation",
            [](Determinant& d, const std::vector<int>& aann, const std::vector<int>& acre,
               const std::vector<int>& bann,
               const std::vector<int>& bcre) { return gen_excitation(d, aann, acre, bann, bcre); },
            "Apply a generic excitation")
        .def("excitation_connection", &Determinant::excitation_connection,
             "Get the excitation connection between this and another determinant")
        .def(
            "str", [](const Determinant& d, int n) { return str(d, n); },
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
        .def("is_empt", &Configuration::is_empt, "n"_a, "Is orbital n empty?")
        .def("is_docc", &Configuration::is_docc, "n"_a, "Is orbital n doubly occupied?")
        .def("is_socc", &Configuration::is_socc, "n"_a, "Is orbital n singly occupied?")
        .def("set_occ", &Configuration::set_occ, "n"_a, "value"_a, "Set the value of an alpha bit")
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