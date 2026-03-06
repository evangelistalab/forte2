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
        .def("set_frozen_creation", &SelectedCIHelper::set_frozen_creation, "frozen_creation"_a,
             "Set orbitals excluded from creation in selection")
        .def("set_screening_criterion", &SelectedCIHelper::set_screening_criterion, "criterion"_a,
             "Set the screening criterion for selection ('hbci' or 'ehbci')")
        .def("set_energy_correction", &SelectedCIHelper::set_energy_correction, "correction"_a,
             "Set the energy correction method for selection ('variational' or 'pt2')")
        .def("select_hbci_ref", &SelectedCIHelper::select_hbci_ref, "var_threshold"_a,
             "pt2_threshold"_a, "Perform HBCI selection with the given threshold")
        .def("select_hbci", &SelectedCIHelper::select_hbci, "var_threshold"_a, "pt2_threshold"_a,
             "Perform HBCI selection with the given thresholds")
        .def("compute_spin2", &SelectedCIHelper::compute_spin2,
             "Compute the expectation value of S^2 for each root and return as a list")
        .def("a_1rdm", &SelectedCIHelper::compute_a_1rdm, "left_root"_a, "right_root"_a,
             "Compute alpha-spin 1-RDM between two roots")
        .def("b_1rdm", &SelectedCIHelper::compute_b_1rdm, "left_root"_a, "right_root"_a,
             "Compute beta-spin 1-RDM between two roots")
        .def("sf_1rdm", &SelectedCIHelper::compute_sf_1rdm, "left_root"_a, "right_root"_a,
             "Compute spin-free 1-RDM between two roots")
        .def("dets", &SelectedCIHelper::variational_dets,
             "Return the determinants in the variational space")
        .def("ndets", &SelectedCIHelper::num_dets_var,
             "Return the number of determinants in the variational space")
        .def("energies", &SelectedCIHelper::energies, "Return the energies of the roots")
        .def("ept2_var", &SelectedCIHelper::ept2_var,
             "Return the variational part of the Epstein-Nesbet second-order energy correction")
        .def("ept2_pt", &SelectedCIHelper::ept2_pt,
             "Return the perturbative part of the Epstein-Nesbet second-order energy correction")
        .def("num_new_dets_var", &SelectedCIHelper::num_new_dets_var,
             "Return the number of new variational determinants added in the last selection")
        .def("num_new_dets_pt2", &SelectedCIHelper::num_new_dets_pt2,
             "Return the number of new perturbative determinants added in the last selection")
        .def("selection_time", &SelectedCIHelper::selection_time,
             "Return the total selection time");
}
} // namespace forte2
