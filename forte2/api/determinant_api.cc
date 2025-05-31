#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "ci/determinant.h"

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
                     } else if (str[i] == '+') {
                         d->set_na(i, true);
                     } else if (str[i] == '-') {
                         d->set_nb(i, true);
                     }
                 }
             })
        // define a static method to create a zero determinant
        .def_static("zero", &Determinant::zero)
        .def("__eq__", [](const Determinant& a, const Determinant& b) { return a == b; })
        .def("__lt__", [](const Determinant& a, const Determinant& b) { return a < b; })
        .def(
            "__repr__", [](Determinant& d) { return str(d); },
            "String representation of the determinant")
        .def("set_na", &Determinant::set_na)
        .def("set_nb", &Determinant::set_nb)
        .def("na", &Determinant::na)
        .def("nb", &Determinant::nb)
        .def("count_a", &Determinant::count_a)
        .def("count_b", &Determinant::count_b)
        .def("count", [](Determinant& d) { return d.count_a() + d.count_b(); });
}
} // namespace forte2