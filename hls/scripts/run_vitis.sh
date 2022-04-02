set -e
FN=forward.mlir make
cp hls_hooks.tcl /home/mlevental/dev_projects/Xilinx/Vitis_HLS/2021.2/common/scripts/hls_hooks.tcl
source /home/mlevental/dev_projects/Xilinx/Vitis_HLS/2021.2/settings64.sh
vitis_hls run_hls.tcl