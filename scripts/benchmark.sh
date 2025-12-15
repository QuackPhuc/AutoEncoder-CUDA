#!/bin/bash
# Benchmark all versions: CPU + GPU (naive, v1, v2, v3)

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
echo "Version,Time_ms,Loss,Memory_GB" > "$RESULTS"

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
    # Parse GPU Memory Used from GPUProfiler output (e.g., "GPU Memory Used:   0.8 GB")
    MEM=$(grep -oP 'GPU Memory Used:\s+\K[0-9.]+' /tmp/bench_out.txt 2>/dev/null | tail -1 || echo "N/A")
    echo "${TIME}ms (${MEM} GB)"
    echo "$name,$TIME,$LOSS,$MEM" >> "$RESULTS"
    eval "${name//-/_}_MS=$TIME"
    eval "${name//-/_}_MEM=$MEM"
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
run_bench "GPU-OptV3" "$GPU_EXE" "--gpu-version 4 --epochs $EPOCHS --samples $SAMPLES"

echo ""

# Summary table
echo "Results:"
printf "  %-12s %10s %12s %10s\n" "Version" "Time(ms)" "Speedup" "Memory(GB)"
echo "  ------------------------------------------------"

if [[ "$GPU_ONLY" == false && -n "${CPU_MS:-}" ]]; then
    printf "  %-12s %10d %12s %10s\n" "CPU" $CPU_MS "1.00x" "N/A"
    BASE=$CPU_MS
else
    BASE=${GPU_Basic_MS:-1}
fi

[[ -n "${GPU_Basic_MS:-}" ]] && printf "  %-12s %10d %12.1fx %10s\n" "GPU-Basic" $GPU_Basic_MS $(echo "scale=1; $BASE/$GPU_Basic_MS" | bc) "${GPU_Basic_MEM:-N/A}"
[[ -n "${GPU_OptV1_MS:-}" ]] && printf "  %-12s %10d %12.1fx %10s\n" "GPU-OptV1" $GPU_OptV1_MS $(echo "scale=1; $BASE/$GPU_OptV1_MS" | bc) "${GPU_OptV1_MEM:-N/A}"
[[ -n "${GPU_OptV2_MS:-}" ]] && printf "  %-12s %10d %12.1fx %10s\n" "GPU-OptV2" $GPU_OptV2_MS $(echo "scale=1; $BASE/$GPU_OptV2_MS" | bc) "${GPU_OptV2_MEM:-N/A}"
[[ -n "${GPU_OptV3_MS:-}" ]] && printf "  %-12s %10d %12.1fx %10s\n" "GPU-OptV3" $GPU_OptV3_MS $(echo "scale=1; $BASE/$GPU_OptV3_MS" | bc) "${GPU_OptV3_MEM:-N/A}"

echo ""
echo "[OK] Saved: $RESULTS"

# Generate chart if Python available
command -v python3 &>/dev/null && python3 scripts/plot_results.py "$RESULTS" 2>/dev/null || true
