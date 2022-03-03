#!/usr/bin/env bash

# This script is an example for running frontier's benchmarks.
# It requires frontier to be compiled with --features=runtime-benchmarks in release mode.

set -e

function choose_and_bench {
    readarray -t options < <(./target/release/frontier-template-node benchmark --list | sed 1d)
    options+=('EXIT')

    select opt in "${options[@]}"; do
        IFS=', ' read -ra parts <<< "${opt}"
        echo "${parts[0]} -- ${parts[1]}"
        [[ "${opt}" == 'EXIT' ]] && exit 0
        
        bench "${parts[0]}" "${parts[1]}"
        break
    done
}

function bench {
    echo "benchmarking ${1}::${2}"
    WASMTIME_BACKTRACE_DETAILS=1 ./target/release/frontier-template-node benchmark \
        --chain dev \
        --execution=wasm \
        --wasm-execution=compiled \
        --pallet "${1}" \
        --extrinsic "${2}" \
        --steps 32 \
        --repeat 64 \
        --template=./benchmarking/frame-weight-template.hbs \
        --record-proof \
        --json-file raw.json \
        --output weights.rs
}

if  [[ $# -eq 1 && "${1}" == "--help" ]]; then
    echo "USAGE:"
    echo "  ${0} [<pallet> <extrinsic>]" 
elif [[ $# -ne 2 ]]; then
    choose_and_bench
else
    bench "${1}" "${2}"
fi
