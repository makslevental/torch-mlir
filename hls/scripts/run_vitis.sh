set -e
#FN=forward.mlir make
cp hls_hooks.tcl /home/maksim/dev_projects/Xilinx/Vitis_HLS/2021.1/common/scripts/hls_hooks.tcl
source /home/maksim/dev_projects/Xilinx/Vitis_HLS/2021.1/settings64.sh
vitis_hls run_hls.tcl
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
cp -R proj "proj-$timestamp"
