from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys
import torch

ROOT = Path(__file__).resolve().parent

lib_dirs = [str(Path(torch.__file__).parent / "lib")]
conda_lib = Path(sys.prefix) / "Library" / "lib"
if conda_lib.exists():
    lib_dirs.append(str(conda_lib))

extra_compile_args = {"cxx": []}

if sys.platform == "win32":
    extra_compile_args["cxx"].append("/std:c++20")
else:
    extra_compile_args["cxx"].append("-std=c++20")

sources = [
    ROOT / "bindings" / "pybind.cpp",
    ROOT / "src" / "tables.cpp",
    ROOT / "src" / "engine.cpp",
    ROOT / "src" / "env.cpp",
    ROOT / "src" / "search.cpp",
]

ext = CppExtension(
    name="bbengine._bbengine",
    sources=[str(s) for s in sources],
    library_dirs=lib_dirs,
    include_dirs=[str(ROOT / "src")],
    extra_compile_args=extra_compile_args,
)

setup(
    name="bbengine",
    version="0.1.0",
    packages=["bbengine"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
