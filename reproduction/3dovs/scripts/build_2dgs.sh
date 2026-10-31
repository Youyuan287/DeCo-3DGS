#!/bin/bash
set -x
B=/root/rivermind-data/deco
V=$B/venv
export TMPDIR=$B/tmp CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.9
export CPATH=/usr/local/cuda-12.1/include:$CPATH
cd $B/2dgs/submodules/diff-surfel-rasterization && $V/bin/pip install -e . --no-build-isolation
cd $B/2dgs/submodules/simple-knn && $V/bin/pip install -e . --no-build-isolation
$V/bin/python -c "from diff_surfel_rasterization import GaussianRasterizer; import simple_knn; print('BUILD_OK')"
