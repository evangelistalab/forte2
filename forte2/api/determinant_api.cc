#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
// #include <nanobind/stl/vector.h>
// #include <nanobind/stl/pair.h>

#include "ci/determinant.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m) {
    nb::class_<Determinant>(m, "Determinant")
        .def("__init__",
             [](Determinant* d, std::string str) {
                 new (d) Determinant{};
                 d->clear();
                 if (str.size() > Determinant::norb) {
                     throw std::runtime_error("Determinant string must be of length " +
                                              std::to_string(Determinant::norb));
                 }
                 for (int i = 0; i < str.size(); ++i) {
                     if (str[i] == '2') {
                         d->set_a(i, true);
                         d->set_b(i, true);
                     } else if (str[i] == '+') {
                         d->set_a(i, true);
                     } else if (str[i] == '-') {
                         d->set_b(i, true);
                     }
                 }
             })
        // define a static method to create a zero determinant
        .def_static("zero", &Determinant::zero)
        .def(nb::init<const Determinant&>())
        .def("__eq__", &Determinant::operator==)
        .def("__repr__", [](Determinant& d) { return str(d); })
        .def("set_a", &Determinant::set_a)
        .def("set_b", &Determinant::set_b)
        .def("get_a", &Determinant::get_a)
        .def("get_b", &Determinant::get_b)
        .def("count_a", &Determinant::count_a)
        .def("count_b", &Determinant::count_b)
        .def("count", &Determinant::count);
}
} // namespace forte2