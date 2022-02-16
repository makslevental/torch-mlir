set -o errexit
set -o pipefail
set -o nounset

# ---------------------- GLOBALS --------------------------------
VITIS_DIR="/home/mlevental/dev_projects/Xilinx_vitis/Vitis_HLS/2021.1"
source "${VITIS_DIR}/settings64.sh"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/build/bin"
ROOT_LIBDIR="${ROOT_DIR}/build/lib"
PROJ_DIR="${DIR}/vitis_stuff"

mkdir -p "${PROJ_DIR}"

export PATH="${ROOT_BINDIR}:${PATH}"

# Get the top-level function name based on the src_file name.
# The rule is simply <name>.c -> "kernel_<name>"
function get_top_func() {
  echo $TOP_FUNC
}

# Do some preprocessing before extracting top function.
function preprocess_mlir() {
  local src_file="${1}"
  local dst_file="${PROJ_DIR}/${src_file%.mlir}.pre.mlir"

  mlir-opt "${src_file}" -canonicalize >"${dst_file}"

  echo "${dst_file}"
}

function lower_mlir_to_llvm() {
  local src_file="${1}"
  local opt_file="${src_file}.opt.mlir"
  local dst_file="${src_file%.mlir}.ll"

  mlir-opt "${src_file}" \
    -lower-affine \
    -convert-scf-to-std \
    -canonicalize \
    -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' \
    -reconcile-unrealized-casts >"${opt_file}"
  mlir-translate -debug -mlir-to-llvmir "${opt_file}" >"${dst_file}"

  echo "${dst_file}"
}

# Call Phism LLVM passes.
function opt_llvm_for_vitis() {
  local src_file="${1}"
  local dst_file="${src_file%.ll}.vitis.ll"
  local top_func
  top_func="$(get_top_func "${src_file}")"

  "${ROOT_BINDIR}/opt" "${src_file}" \
    -S \
    -enable-new-pm=0 \
    -load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
    -mem2arr \
    -strip-debug \
    -instcombine \
    -xlnanno -xlntop="${top_func}" \
    -xlnmath \
    -xlnname \
    -strip-attr \
    -xlnunroll \
    -xlnarraypartition \
    > "${dst_file}"

  echo "${dst_file}"
}

# Generate dummy C source.
function gen_dummy_c() {
  local src_file="${1}"
  local src_base
  local top_func
  local dst_file

  src_base="$(basename "$(dirname "${src_file}")")"
  dst_file="$(dirname "${src_file}")/${src_base}.dummy.c"
  top_func="$(get_top_func "${src_file}")"

  cat <<EOF >"${dst_file}"
void ${top_func}_dummy() {}
EOF

  echo "${dst_file}"
}

# Generate run_hls.tcl that passes Phism generated LLVM-IR to Vitis.
# Returns the full file path of run_hls.tcl
function gen_vitis_phism_tcl() {
  local src_file="${1}"
  local src_base
  local src_dir
  local dst_file
  local top_func
  local dummy_c_src

  src_dir="$(dirname "${src_file}")"
  src_base="$(basename "${src_dir}")"
  dst_file="$(dirname "${src_file}")/run_hls.tcl"
  top_func="$(get_top_func "${src_file}")"
#  dummy_c_src="$(gen_dummy_c "${src_file}")"

  # Synthesis script
  cat <<EOF >"${dst_file}"
open_project -reset proj
#add_files /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/vitis_stuff.dummy.c
add_files /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/wrapper.cpp
set_top wrapper

open_solution -reset solution1
#set_part "xcvu35p-fsvh2104-3-e"
set_part "xc7a200tfbg484-3"
create_clock -period "100MHz"
config_export -format ip_catalog -rtl verilog

set ::LLVM_CUSTOM_CMD {\$LLVM_CUSTOM_OPT ${src_file} -o \$LLVM_CUSTOM_OUTPUT}

csynth_design

export_design -rtl verilog -format ip_catalog

EOF

  echo "${dst_file}"
}

# Run Vitis.
function run_vitis() {
  local src_file="${1}"
  local phism_tcl_file
  local src_dir

  src_dir="$(dirname "${src_file}")"
  phism_tcl_file="$(gen_vitis_phism_tcl "${src_file}")"

  cd "${src_dir}"

  # Synthesize for Phism
  vitis_hls "${phism_tcl_file}" &>"${src_dir}"/vhls.syn.log

  # Read time from vitis_hls.log
  # e.g. $finish called at time : 13920290 ps
  # JC: there is a formula to convert this to cycles, but I do not remember now - it should be OK for now

  local status=$?

  cd - >/dev/null

  echo "${status}"
}

# Evaluate a single C source file. This is the Phism pipeline btw.
function eval_file() {
  local src_file="${1}"

  printf ">> Evaluating source file: %s ..." "${src_file}"

  for skipped_example in "${SKIPPED_EXAMPLES[@]}"; do
    if [[ "${src_file}" == *"${skipped_example}"* ]]; then
      return
    fi
  done

  prep_src_file="$(preprocess_mlir "${src_file}")"
  llvm_src_file="$(lower_mlir_to_llvm "${prep_src_file}")"
  vitis_src_file="$(opt_llvm_for_vitis "${llvm_src_file}")"

  local status
  status="$(run_vitis "${vitis_src_file}")"

  if [[ "${status}" == "0" ]]; then
    echo " SUCCESS"
  else
    echo " FAILED"
  fi
}

# export TOP_FUNC=matmul
export TOP_FUNC=forward
#eval_file matmul.llvm.mlir
eval_file braggnn.llvm.mlir
