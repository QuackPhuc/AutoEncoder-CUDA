#!/bin/bash
# Benchmark all versions: CPU + GPU (naive, v1, v2)

set -e

EPOCHS=1
SAMPLES=1000
GPU_ONLY=false

show_help() {
    cat << 'EOF'
Usage: ./scripts/benchmark.sh [OPTIONS]

Benchmark performance across CPU and GPU versions

OPTIONS:
    --epochs N      Training epochs (default: 1)
    --samples N     Sample count (default: 1000)
    --gpu-only      Skip CPU benchmark
    --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --gpu-only) GPU_ONLY=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p ./results
RESULTS="./results/benchmark.csv"
echo "Version,Time_ms,Loss" > "$RESULTS"

echo "[benchmark] epochs=$EPOCHS samples=$SAMPLES"
echo ""

# Helper: run and time
run_bench() {
    local name="$1" exe="$2" args="$3"
    echo -n "  $name... "
    START=$(date +%s%3N)
    $exe $args > /tmp/bench_out.txt 2>&1 || true
    END=$(date +%s%3N)
    TIME=$((END - START))
    LOSS=$(grep -oP 'Loss: \K[0-9.]+' /tmp/bench_out.txt 2>/dev/null | tail -1 || echo "N/A")
    echo "${TIME}ms"
    echo "$name,$TIME,$LOSS" >> "$RESULTS"
    eval "${name//-/_}_MS=$TIME"
}

# Find executables
CPU_EXE="./build/bin/autoencoder_cpu"
GPU_EXE="./build/bin/autoencoder_gpu"
[[ -f "${CPU_EXE}.exe" ]] && CPU_EXE="${CPU_EXE}.exe"
[[ -f "${GPU_EXE}.exe" ]] && GPU_EXE="${GPU_EXE}.exe"

# Run benchmarks
[[ "$GPU_ONLY" == false ]] && run_bench "CPU" "$CPU_EXE" "--epochs $EPOCHS --samples $SAMPLES"
run_bench "GPU-Basic" "$GPU_EXE" "--gpu-version 1 --epochs $EPOCHS --samples $SAMPLES"
run_bench "GPU-OptV1" "$GPU_EXE" "--gpu-version 2 --epochs $EPOCHS --samples $SAMPLES"
run_bench "GPU-OptV2" "$GPU_EXE" "--gpu-version 3 --epochs $EPOCHS --samples $SAMPLES"

echo ""

# Summary table
echo "Results:"
printf "  %-12s %10s %12s\n" "Version" "Time(ms)" "Speedup"
echo "  ------------------------------------"

if [[ "$GPU_ONLY" == false && -n "${CPU_MS:-}" ]]; then
    printf "  %-12s %10d %12s\n" "CPU" $CPU_MS "1.00x"
    BASE=$CPU_MS
else
    BASE=${GPU_Basic_MS:-1}
fi

[[ -n "${GPU_Basic_MS:-}" ]] && printf "  %-12s %10d %12.1fx\n" "GPU-Basic" $GPU_Basic_MS $(echo "scale=1; $BASE/$GPU_Basic_MS" | bc)
[[ -n "${GPU_OptV1_MS:-}" ]] && printf "  %-12s %10d %12.1fx\n" "GPU-OptV1" $GPU_OptV1_MS $(echo "scale=1; $BASE/$GPU_OptV1_MS" | bc)
[[ -n "${GPU_OptV2_MS:-}" ]] && printf "  %-12s %10d %12.1fx\n" "GPU-OptV2" $GPU_OptV2_MS $(echo "scale=1; $BASE/$GPU_OptV2_MS" | bc)

echo ""
echo "[OK] Saved: $RESULTS"

# Generate chart if Python available
command -v python3 &>/dev/null && python3 scripts/plot_results.py "$RESULTS" 2>/dev/null || true
