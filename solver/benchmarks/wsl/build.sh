#!/usr/bin/env bash
# Builds SPRAL (SSIDS, CPU only) and the ssids_replay driver inside WSL2
# Ubuntu. Idempotent. Output stays in $HOME of the WSL user (never committed):
#   $HOME/spral-install/            SPRAL install (lib/libspral.so, include/)
#   $HOME/ssids_bench/ssids_replay  the driver run_wsl.py launches
# Prerequisites (apt): gfortran gcc g++ meson ninja-build libmetis-dev
#   libopenblas-dev libhwloc-dev pkg-config curl
# Usage (from Windows):
#   wsl.exe -d Ubuntu -- bash /mnt/c/<repo>/solver/benchmarks/wsl/build.sh
set -euo pipefail
SPRAL_VERSION="${SPRAL_VERSION:-v2025.09.18}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$HOME/spral-install"

if [ ! -f "$PREFIX/lib/libspral.so" ]; then
    mkdir -p "$HOME/src" && cd "$HOME/src"
    if [ ! -d spral ]; then
        curl -sL "https://github.com/ralna/spral/archive/refs/tags/${SPRAL_VERSION}.tar.gz" | tar xz
        mv "spral-${SPRAL_VERSION#v}" spral
    fi
    cd spral
    [ -d builddir ] || meson setup builddir --prefix="$PREFIX" \
        -Dgpu=false -Dopenmp=true -Dexamples=false -Dtests=false -Dbuildtype=release
    meson compile -C builddir
    meson install -C builddir
fi

mkdir -p "$HOME/ssids_bench"
gcc -O2 -std=gnu11 -fopenmp -o "$HOME/ssids_bench/ssids_replay" "$HERE/ssids_replay.c" \
    -I"$PREFIX/include" -L"$PREFIX/lib" -lspral -lm -Wl,-rpath,"$PREFIX/lib"
echo "built $HOME/ssids_bench/ssids_replay"
