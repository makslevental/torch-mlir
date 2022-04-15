set -e
#FN=forward.mlir make
source /home/$USER/dev_projects/Xilinx/Vitis_HLS/2021.2/settings64.sh
vivado -mode tcl -stack 2000 -source run_vivado.tcl
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
cp -R proj "proj-$timestamp"
