#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 URL OUTPUT EXPECTED_BYTES [PARTS]" >&2
    exit 2
fi

url="$1"
output="$2"
expected_bytes="$3"
parts="${4:-8}"

if [[ -z "$url" || -z "$output" || ! "$expected_bytes" =~ ^[0-9]+$ || ! "$parts" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid URL, output path, size, or part count." >&2
    exit 2
fi

part_dir="${output}.parts"
mkdir -p "$part_dir"
chunk_bytes=$(( (expected_bytes + parts - 1) / parts ))
pids=()

for ((index = 0; index < parts; index++)); do
    start=$(( index * chunk_bytes ))
    end=$(( start + chunk_bytes - 1 ))
    (( end >= expected_bytes )) && end=$(( expected_bytes - 1 ))
    part="${part_dir}/part_$(printf '%03d' "$index")"
    part_bytes=$(( end - start + 1 ))

    (
        if [[ -f "$part" ]] && [[ "$(stat -c '%s' "$part")" -eq "$part_bytes" ]]; then
            echo "Part $index already complete."
            exit 0
        fi

        echo "Downloading part $index: bytes $start-$end"
        curl --location --fail --retry 10 --retry-all-errors \
            --connect-timeout 30 --range "$start-$end" \
            "$url" --output "${part}.tmp"

        actual_bytes="$(stat -c '%s' "${part}.tmp")"
        if [[ "$actual_bytes" -ne "$part_bytes" ]]; then
            echo "Part $index has $actual_bytes bytes; expected $part_bytes." >&2
            exit 1
        fi
        mv "${part}.tmp" "$part"
    ) &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

assembled="${output}.assembled"
: > "$assembled"
for ((index = 0; index < parts; index++)); do
    part="${part_dir}/part_$(printf '%03d' "$index")"
    cat "$part" >> "$assembled"
done

actual_bytes="$(stat -c '%s' "$assembled")"
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "Assembled file has $actual_bytes bytes; expected $expected_bytes." >&2
    exit 1
fi

mv "$assembled" "$output"
rm -r "$part_dir"
echo "Downloaded $output ($actual_bytes bytes)."
