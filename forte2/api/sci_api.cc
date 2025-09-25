#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>

#include "ci/sci_helper.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sci_helper_api(nb::module_& m) {
    nb::class_<SelectedCIHelper>(m, "SelectedCIHelper")
        .def(nb::init<size_t, const std::vector<Determinant>&, np_matrix&, double, np_matrix&,
                      np_tensor4&, int>(),
             "norb"_a, "dets"_a, "c"_a, "E"_a, "H"_a, "V"_a, "log_level"_a = 3,
             "Initialize the SelectedCIHelper with the number of orbitals, initial determinants, "
             "energy, Hamiltonian, and integrals")
        .def("set_Hamiltonian", &SelectedCIHelper::set_Hamiltonian, "E"_a, "H"_a, "V"_a,
             "Set the Hamiltonian integrals")
        .def("set_c", &SelectedCIHelper::set_c, "c"_a, "Set the CI coefficients")
        .def("select_cipsi", &SelectedCIHelper::select_cipsi, "threshold"_a,
             "Perform CIPSI selection with the given threshold")
        .def("dets", &SelectedCIHelper::get_variational_dets,
             "Return the determinants in the variational space")
        .def(
            "ndets", [](SelectedCIHelper& self) { return self.get_variational_dets().size(); },
            "Return the number of determinants in the variational space");
}
} // namespace forte2
