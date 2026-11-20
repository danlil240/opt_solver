# BinaryBuilder.jl recipe for smf_jll (Yggdrasil: S/smf/build_tarballs.jl)
using BinaryBuilder, Pkg

name = "smf"
version = v"0.1.0"

sources = [
    GitSource("https://github.com/danlil240/opt_solver.git",
              "REPLACE_WITH_RELEASE_COMMIT_SHA"),
]

script = raw"""
cd $WORKSPACE/srcdir/opt_solver

MARCH_OPT="-DSMF_MARCH="
if [[ "${target}" == x86_64-* ]]; then
    MARCH_OPT="-DSMF_MARCH=x86-64-v3"
fi

cmake -S solver -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DSMF_USE_MKL=OFF \
    -DSMF_USE_SUITESPARSE_AMD=ON \
    -DSMF_USE_METIS=ON \
    -DSMF_VENDORED_METIS=ON \
    -DSMF_BUILD_C_API=ON \
    -DSMF_BUILD_MA97_PLUGIN=ON \
    -DSMF_BUILD_TESTS=OFF \
    -DSMF_BUILD_BENCHMARKS=OFF \
    -DBLAS_LIBRARIES="-L${libdir} -lopenblas" \
    -DLAPACK_LIBRARIES="-L${libdir} -lopenblas" \
    ${MARCH_OPT}

cmake --build build -j ${nproc}
cmake --install build

install_license LICENSE
install_license solver/third_party/LICENSE-Apache-2.0.txt
"""

platforms = expand_cxxstring_abis(supported_platforms(; exclude=p -> arch(p) == "i686"))

products = [
    LibraryProduct("libsmf_c", :libsmf_c),
    LibraryProduct("libsmf_ma97", :libsmf_ma97),
]

dependencies = [
    Dependency("OpenBLAS32_jll"),
    Dependency("SuiteSparse32_jll"),
    Dependency("CompilerSupportLibraries_jll"),
    Dependency("LLVMOpenMP_jll"; platforms=filter(Sys.isapple, platforms)),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.10", preferred_gcc_version=v"11")
