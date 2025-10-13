#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>

#include "sci/sci_helper.h"

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
        .def("Hamiltonian", &SelectedCIHelper::Hamiltonian, "basis"_a, "sigma"_a,
             "Apply the Hamiltonian to the basis and store the result in sigma")
        .def("Hdiag", &SelectedCIHelper::Hdiag, "Return the diagonal of the Hamiltonian matrix")
        .def("set_c", &SelectedCIHelper::set_c, "c"_a, "Set the CI coefficients")
        .def("set_num_threads", &SelectedCIHelper::set_num_threads, "n"_a,
             "Set the number of threads to use in parallel sections")
        .def("set_num_batches_per_thread", &SelectedCIHelper::set_num_batches_per_thread, "n"_a,
             "Set the number of batches each thread will process in parallel sections")
        .def("set_energies", &SelectedCIHelper::set_energies, "e"_a,
             "Set the energies of the roots")
        .def("select_cipsi", &SelectedCIHelper::select_cipsi, "threshold"_a,
             "Perform CIPSI selection with the given threshold")
        .def("select_hbci_ref", &SelectedCIHelper::select_hbci_ref, "var_threshold"_a,
             "pt2_threshold"_a, "Perform HBCI selection with the given threshold")
        .def("select_hbci2", &SelectedCIHelper::select_hbci2, "var_threshold"_a, "pt2_threshold"_a,
             "Perform HBCI2 selection with the given thresholds")
        .def("select_hbci3", &SelectedCIHelper::select_hbci3, "var_threshold"_a, "pt2_threshold"_a,
             "Perform HBCI3 selection with the given thresholds")
        .def("compute_spin2", &SelectedCIHelper::compute_spin2,
             "Compute the expectation value of S^2 for each root and return as a list")
        .def("dets", &SelectedCIHelper::get_variational_dets,
             "Return the determinants in the variational space")
        .def(
            "ndets", [](SelectedCIHelper& self) { return self.get_variational_dets().size(); },
            "Return the number of determinants in the variational space")
        .def("get_energies", &SelectedCIHelper::get_energies, "Return the energies of the roots")
        .def("get_ept2_var", &SelectedCIHelper::get_ept2_var,
             "Return the variational part of the Epstein-Nesbet second-order energy correction")
        .def("get_ept2_pt", &SelectedCIHelper::get_ept2_pt,
             "Return the perturbative part of the Epstein-Nesbet second-order energy correction")
        .def("get_num_new_dets_var", &SelectedCIHelper::get_num_new_dets_var,
             "Return the number of new variational determinants added in the last selection")
        .def("get_num_new_dets_pt2", &SelectedCIHelper::get_num_new_dets_pt2,
             "Return the number of new perturbative determinants added in the last selection")
        .def("get_selection_time", &SelectedCIHelper::get_selection_time,
             "Return the total selection time");
}
} // namespace forte2
