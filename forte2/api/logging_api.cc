#include <nanobind/nanobind.h>
#include "helpers/logger.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {
void export_logging_api(nb::module_& m) {
    m.def(
        "set_log_level",
        [](int level) { Logger::getInstance().setLevel(static_cast<Logger::Level>(level)); },
        "Set the logging verbosity level (0=NONE, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)");
    m.def(
        "get_log_level", []() { return static_cast<int>(Logger::getInstance().getLevel()); },
        "Get the current logging verbosity level");
}
} // namespace forte2