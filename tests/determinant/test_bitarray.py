import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_bitarray_subscript_lifetime(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a C++ compiler is required to test BitArray subscripting")

    source = """
#include <cassert>
#include <type_traits>
#include <utility>
#include "determinant/determinant.h"

template <typename Bits> void check_subscript() {
    static_assert(std::is_same_v<decltype(Bits::zero()[0]), bool>);
    static_assert(std::is_same_v<decltype(std::declval<const Bits&&>()[0]), bool>);
    static_assert(std::is_same_v<decltype(std::declval<const Bits&>()[0]), bool>);
    static_assert(std::is_same_v<decltype(std::declval<Bits&>()[0]), typename Bits::Proxy>);

    const auto make_occupied = [] {
        auto bits = Bits::zero();
        bits[0] = true;
        return bits;
    };
    const auto value = make_occupied()[0];
    assert(value);
    assert(!Bits::zero()[0]);
    const auto bits = make_occupied();
    assert(bits[0]);
}

int main() {
    check_subscript<forte2::BitArray<64>>();
    check_subscript<forte2::String>();
    check_subscript<forte2::Determinant>();
}
"""
    include_dir = Path(__file__).resolve().parents[2] / "forte2"
    executable = tmp_path / "bitarray_subscript"
    command = compiler + [
        "-std=c++23", "-I", str(include_dir), "-x", "c++", "-", "-o", str(executable)
    ]
    result = subprocess.run(command, input=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(executable)], check=True)
