#!/usr/bin/env bash

set -xeu -o pipefail

SRC_ROOT="$( cd "$(dirname "$0")" ; pwd -P)/.."
PYTORCH_ROOT=${PYTORCH_ROOT:-$SRC_ROOT/externals/pytorch}
PYTORCH_INSTALL_PATH=${PYTORCH_INSTALL_PATH:-$SRC_ROOT/libtorch}
PYTORCH_BRANCH="${PYTORCH_BRANCH:-master}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-shared}"
PT_C_COMPILER="${PT_C_COMPILER:-clang}"
PT_CXX_COMPILER="${PT_CXX_COMPILER:-clang++}"
WHEELHOUSE="${WHEELHOUSE:-$SRC_ROOT/build_tools/python_deploy/wheelhouse}"

Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[1;33m'
NC='\033[0m'

echo "SRC_ROOT=${SRC_ROOT}"
echo "PYTORCH_ROOT=${PYTORCH_ROOT}"
echo "PYTORCH_BRANCH=${PYTORCH_BRANCH}"
echo "LIBTORCH_VARIANT=${LIBTORCH_VARIANT}"
echo "LIBTORCH_SRC_BUILD=${LIBTORCH_SRC_BUILD}"
echo "LIBTORCH_CACHE=${LIBTORCH_CACHE}"
echo "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
export CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
export CMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
export CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}

if [[ "$LIBTORCH_VARIANT" == *"cxx11-abi"* ]]; then
  echo _GLIBCXX_USE_CXX11_ABI=1
  export _GLIBCXX_USE_CXX11_ABI=1
  CXX_ABI=1
else
  echo _GLIBCXX_USE_CXX11_ABI=0
  export _GLIBCXX_USE_CXX11_ABI=0
  CXX_ABI=0
fi

retry () {
  "$@" || (sleep 1 && "$@") || (sleep 2 && "$@") || (sleep 4 && "$@") || (sleep 8 && "$@")
}

install_requirements() {
  pip install -qr "$PYTORCH_ROOT/requirements.txt"
  pip list
}

# Check for an existing libtorch at $PYTORCH_ROOT
check_existing_libtorch() {
  if [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.so" ]]; then
    echo "Existing PYTORCH shared build found.. skipping build"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.a" ]]; then
    echo "Existing PYTORCH static build found.. skipping build"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.dylib" ]]; then
    echo "Existing PYTORCH shared dylib found.. skipping build"
    return 0
  fi
  return 1
}

# Download and unzip into externals/pytorch/libtorch
MACOS_X86_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip"
# Download here (Pre-cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
# Download here (cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
# Download static here (cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-static-with-deps-latest.zip
# static builds are broken upstream and ship shared libraries anyway. Hopefully we can reland the fix upstream.
LINUX_X86_URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-static-without-deps-latest.zip"

download_libtorch() {
  cd "$SRC_ROOT"
  if [[ $(uname -s) = 'Darwin' ]]; then
  echo "Apple macOS detected"
  if [[ $(uname -m) == 'arm64' ]]; then
    echo "${Red}Apple M1 Detected...no libtorch/ binaries available${NC}"
    return 1
  else
    echo "Apple x86_64 Detected"
    DOWNLOAD_URL=${MACOS_X86_URL}
  fi
elif [[ $(uname -s) = 'Linux' ]]; then
  echo "Linux detected"
  DOWNLOAD_URL=${LINUX_X86_URL}
else
  echo "OS not detected. Pray and Play"
  return 1
fi
  echo "Deleting any old libtorch*.."
  rm -rf libtorch*
  curl -O ${DOWNLOAD_URL}
  unzip -q -o libtorch-*.zip
  rm libtorch-*.zip
  if [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.so" ]]; then
    echo "Verifying Pytorch install -- libtorch.so found"
    return 0
  elif [[ -f "$PYTORCH_INSTALL_PATH/lib/libtorch.a" ]]; then
    echo "Verifying Pytorch install -- libtorch.a found"
    return 0
  fi
  return 1
}

checkout_pytorch() {
  if [[ ! -d "$PYTORCH_ROOT" ]]; then
    git clone --depth 1 --single-branch --branch "${PYTORCH_BRANCH}" https://github.com/pytorch/pytorch "$PYTORCH_ROOT"
  fi
  cd "$PYTORCH_ROOT"
  for dep in protobuf pocketfft cpuinfo FP16 psimd fmt sleef pybind11 onnx flatbuffers foxi; do
    git submodule update --init --depth 1 -- third_party/$dep
  done
}

build_pytorch() {
  cd "$PYTORCH_ROOT"
  # Uncomment the next line if you want to iterate on source builds
  # python setup.py clean

  sed -i.bak -E 's/^[[:space:]]+check_submodules()/#check_submodules()/g' setup.py
  # remove after https://github.com/pytorch/pytorch/issues/81178 is fixed
  sed -i.bak -E 's/namespace ao {/namespace ao {\n#include <ATen\/native\/ao_sparse\/quantized\/cpu\/packed_params.h>/g' aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_deserialize.cpp
  sed -i.bak -E 's/namespace ao {/namespace ao {\n#include <ATen\/native\/ao_sparse\/quantized\/cpu\/packed_params.h>/g' aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_serialize.cpp

  if ! command -v ninja &> /dev/null
  then
    echo "You should probably install Ninja using pip install ninja"
    USE_NINJA=0
    MAX_JOBS=1
  else
    USE_NINJA=1
    if [[ -z ${MAX_JOBS+""} ]]; then
      if [[ "$(uname -s)" == 'Darwin' ]]; then
        MAX_JOBS=$(sysctl -n hw.ncpu)
      else
        MAX_JOBS=$(nproc)
      fi
    fi
  fi

  BUILD_SHARED_LIBS=ON
  USE_LIGHTWEIGHT_DISPATCH=OFF
  STATIC_DISPATCH_BACKEND=OFF
  BUILD_LITE_INTERPRETER=OFF
  if [[ $LIBTORCH_VARIANT = *"static"* ]]; then
    BUILD_SHARED_LIBS=OFF
    # Enable after more testing.
    # USE_LIGHTWEIGHT_DISPATCH=ON
    # STATIC_DISPATCH_BACKEND=ON
    # BUILD_LITE_INTERPRETER=OFF
  fi
  BUILD_LITE_INTERPRETER=${BUILD_LITE_INTERPRETER} \
  BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
  BUILD_TEST=OFF \
  CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI}" \
  CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES} \
  MAX_JOBS=${MAX_JOBS} \
  ONNX_ML=OFF \
  STATIC_DISPATCH_BACKEND=${STATIC_DISPATCH_BACKEND} \
  USE_BREAKPAD=OFF \
  USE_CUDA=OFF \
  USE_DISTRIBUTED=OFF \
  USE_EIGEN_FOR_BLAS=OFF \
  USE_FBGEMM=OFF \
  USE_GLOO=OFF \
  USE_KINETO=OFF \
  USE_LIGHTWEIGHT_DISPATCH=${USE_LIGHTWEIGHT_DISPATCH} \
  USE_MKL=OFF \
  USE_MKLDNN=OFF \
  USE_MPS=OFF \
  USE_NCCL=OFF \
  USE_NINJA=${USE_NINJA} \
  USE_NNPACK=OFF \
  USE_OBSERVERS=OFF \
  USE_OPENMP=OFF \
  USE_PYTORCH_QNNPACK=OFF \
  USE_QNNPACK=OFF \
  USE_XNNPACK=OFF \
  python setup.py bdist_wheel -d "$WHEELHOUSE"
}

package_pytorch() {
  rm -rf libtorch/{lib,bin,include,share}

  # Copy over all of the cmake files
  mv build/lib*/torch/share     libtorch/
  mv build/lib*/torch/include   libtorch/
  mv build/lib*/torch/lib       libtorch/
  # Copy over all lib files
  mv build/lib/*                libtorch/lib/
  # Copy over all include files
  mv build/include/*            libtorch/include/

  (pushd "$PYTORCH_ROOT" && git rev-parse HEAD) > libtorch/build-hash
  echo "Installing libtorch in ${PYTORCH_ROOT}/../../"
  echo "deleting old ${PYTORCH_ROOT}/../../libtorch"
  rm -rf "${PYTORCH_ROOT}"/../../libtorch
  mv libtorch "${PYTORCH_ROOT}"/../../
}

#main
if [[ $LIBTORCH_SRC_BUILD = "ON" ]]; then
  echo "Building libtorch from source"
  checkout_pytorch
  install_requirements
  build_pytorch
  package_pytorch
else
  if check_existing_libtorch; then
    echo "Found existing libtorch"
    if [[ $LIBTORCH_CACHE = "ON" ]]; then
      echo "${Yellow} libtorch is being cached. If you have a different PyTorch version pip installed unset LIBTORCH_CACHE ${NC}"
    else
      echo "${Red}Updating libtorch to the latest version. Set -DLIBTORCH_CACHE=ON to prevent autoupdating ${NC}"
      echo "Downloading libtorch..."
      download_libtorch
    fi
  else
    echo "${Green}Installing latest version of libtorch. Set -DLIBTORCH_CACHE=ON in your next run to prevent autoupdating ${NC}"
    echo "Downloading libtorch..."
    download_libtorch
  fi
fi
