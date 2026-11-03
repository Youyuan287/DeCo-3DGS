#!/usr/bin/env bash
# 应用复现所需的 3 处兼容性补丁（在上游仓库根目录执行）。
# 上游源码保持不动；本脚本仅在你需要实际跑通评测时使用。
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
for p in 01_merge_proj_dtype 02_spt_keys_shim 03_setup_cuda_include; do
  echo "==> $p"
  git apply --check "reproduction/patches/${p}.diff" 2>/dev/null \
    && git apply "reproduction/patches/${p}.diff" \
    && echo "    applied" \
    || echo "    已应用或上下文不匹配，跳过（用 git apply --3way 手动处理）"
done
echo "done. 提示：03 号补丁涉及子模块目录改名，如已编译过可忽略。"
