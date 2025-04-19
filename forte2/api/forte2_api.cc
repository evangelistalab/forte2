#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

#include "integrals_api.h"

NB_MODULE(_forte2, m) { forte2::export_integrals_api(m); }