#!/usr/bin/env bash
set -euo pipefail
root=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/runs/readout_20261024/figurines
source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features
manifest=/root/rivermind-data/ARIS-ReLaGS/logs/dedup_maps_20261024.tsv
test "$(realpath "$root")" = "$root"
test "$(realpath "$source")" = "$source"
test ! -e "$manifest"
printf 'target\tsource\tbytes\tsha256\n' > "$manifest"
count=0
bytes=0
for name in r1_dense r1_dense_fullresize r2_min r2_min2; do
  directory="$root/$name"
  test "$(realpath "$directory")" = "$directory"
  for target in "$directory"/*_s.npy; do
    test -f "$target"
    test ! -L "$target" || continue
    original="$source/$(basename "$target")"
    test -f "$original"
    if ! cmp -s "$target" "$original"; then
      printf 'DIFFERENT retained: %s\n' "$target"
      continue
    fi
    size=$(stat -c %s "$target")
    hash=$(sha256sum "$target")
    hash=${hash%% *}
    printf '%s\t%s\t%s\t%s\n' "$target" "$original" "$size" "$hash" >> "$manifest"
    test ! -e "$target.dedup-link"
    ln -s "$original" "$target.dedup-link"
    mv -T "$target.dedup-link" "$target"
    test -L "$target"
    cmp -s "$target" "$original"
    count=$((count+1))
    bytes=$((bytes+size))
  done
done
printf 'replaced=%s logical_bytes_released=%s\n' "$count" "$bytes"
df -h /root/rivermind-data
