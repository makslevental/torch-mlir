set -e
#FN=forward.mlir make
cp hls_hooks.tcl /home/$USER/dev_projects/Xilinx/Vitis_HLS/2021.2/common/scripts/hls_hooks.tcl
source /home/$USER/dev_projects/Xilinx/Vitis_HLS/2021.2/settings64.sh
vitis_hls run_hls.tcl
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
cp -R proj "proj-$timestamp"
