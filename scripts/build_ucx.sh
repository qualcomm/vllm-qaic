#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

set -euxo pipefail

#set path where to install ucx
PREFIX="/usr/local"
# Install dependencies (Ubuntu)

sudo apt update
sudo apt install -y build-essential autoconf \
     automake libtool pkg-config rdma-core libibverbs-dev \
     librdmacm-dev ibverbs-utils git

# clone open source ucx library from git
git clone https://github.com/openucx/ucx.git
cd ucx

# run autogen
./autogen.sh

# make the build folder
mkdir -p build
cd build

# Determine whether RDMA/IB support is available
UCX_CONFIG_FLAGS=(
    "--prefix=${PREFIX}"
    "--enable-mt"
)

if [ -d /dev/infiniband ] \
   && pkg-config --exists libibverbs \
   && pkg-config --exists librdmacm; then

    echo "RDMA/IB detected. Building UCX with verbs support."
    UCX_CONFIG_FLAGS+=("--with-verbs")

else
    echo "WARNING: RDMA/IB support not detected."
    echo "WARNING: Building UCX without InfiniBand/RDMA support."
    echo "WARNING: To enable RDMA support, install:"
    echo "         rdma-core libibverbs-dev librdmacm-dev"
fi

# configure with multi-threading enabled for ucx release
../contrib/configure-release "${UCX_CONFIG_FLAGS[@]}"

# build
make -j$(nproc)

# install
sudo make install
sudo ldconfig

# check ucx version
echo "UCX version"
ucx_info -v

# check ucx devices
ucx_info -d | grep -E "mlx5|verbs|rc_|dc_|ud_|tcp|posix" || true