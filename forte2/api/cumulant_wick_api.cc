#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include "sparse/cumulant_reference.h"
#include "sparse/cumulant_wick.h"
#include "sparse/sparse_generalized_normal_order_product.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_cumulant_wick_api(nb::module_& m) {
    nb::class_<CumulantReference>(
        m, "CumulantReference",
        "Spin-orbital density cumulants for a sparse multiconfigurational reference")
        .def(nb::init<const SparseState&, std::size_t, int, double>(), "vacuum"_a, "norb"_a,
             "max_cumulant"_a = 2, "screen_thresh"_a = 1.0e-12)
        .def("vacuum", &CumulantReference::vacuum, "Get the source sparse reference state")
        .def("norb", &CumulantReference::norb, "Get the number of spatial orbitals")
        .def("max_cumulant", &CumulantReference::max_cumulant,
             "Get the largest available cumulant rank")
        .def("screen_thresh", &CumulantReference::screen_thresh, "Get the construction threshold")
        .def("core_modes", &CumulantReference::core_modes,
             "Get the fixed occupied spin-orbital mask")
        .def("active_modes", &CumulantReference::active_modes,
             "Get the variably occupied spin-orbital mask")
        .def("virtual_modes", &CumulantReference::virtual_modes,
             "Get the unoccupied spin-orbital mask")
        .def("gamma", &CumulantReference::gamma, "p"_a, "p_alpha"_a, "q"_a, "q_alpha"_a,
             "Return gamma^p_q = <a^+_p a_q>")
        .def("eta", &CumulantReference::eta, "p"_a, "p_alpha"_a, "q"_a, "q_alpha"_a,
             "Return eta^p_q = delta^p_q - gamma^p_q")
        .def("rdm", &CumulantReference::rdm, "cre"_a, "ann"_a,
             "Return an RDM element encoded by determinant bit strings")
        .def("truncated_rdm", &CumulantReference::truncated_rdm, "cre"_a, "ann"_a,
             "Return an RDM element reconstructed with unavailable cumulants set to zero")
        .def("cumulant", &CumulantReference::cumulant, "cre"_a, "ann"_a,
             "Return a density cumulant encoded by determinant bit strings")
        .def("cumulant_size", &CumulantReference::cumulant_size, "rank"_a,
             "Get the number of explicitly stored nonzero cumulants at a rank");

    nb::class_<GeneralizedNormalOrderedProductComputer>(
        m, "GeneralizedNormalOrderedProductComputer",
        "Sparse generalized-normal-ordered products evaluated through bare operator strings")
        .def(nb::init<int, double>(), "max_rank"_a, "screen_thresh"_a = 1.0e-12,
             "Use the legacy density-moment rank truncation")
        .def(nb::init<const CumulantReference&, int, double>(), "reference"_a, "max_rank"_a,
             "screen_thresh"_a = 1.0e-12,
             "Reconstruct higher moments from the available density cumulants")
        .def("uses_cumulant_truncation",
             &GeneralizedNormalOrderedProductComputer::uses_cumulant_truncation,
             "Whether unavailable density cumulants are set to zero")
        .def("commutator", &GeneralizedNormalOrderedProductComputer::commutator, "lhs"_a, "rhs"_a,
             "Compute a generalized-normal-ordered commutator");

    nb::class_<CumulantWickEngine>(
        m, "CumulantWickEngine",
        "Direct generalized Wick products using explicit spin-orbital density cumulants")
        .def(nb::init<const CumulantReference&, int, double>(), "reference"_a, "max_rank"_a,
             "screen_thresh"_a = 1.0e-12)
        .def("reference", &CumulantWickEngine::reference, "Get the explicit cumulant reference")
        .def("max_rank", &CumulantWickEngine::max_rank, "Get the maximum retained many-body rank")
        .def("screen_thresh", &CumulantWickEngine::screen_thresh,
             "Get the numerical screening threshold")
        .def("product", &CumulantWickEngine::product, "lhs"_a, "rhs"_a,
             "Compute a direct generalized-normal-ordered product with unavailable cumulants set "
             "to zero")
        .def("commutator", &CumulantWickEngine::commutator, "lhs"_a, "rhs"_a,
             "Compute a direct generalized-normal-ordered commutator with unavailable cumulants "
             "set to zero");
}

} // namespace forte2
