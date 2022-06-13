set -e
#FN=forward.mlir make
rm -rf project_1
source /home/$USER/Xilinx/Vitis_HLS/2021.2/settings64.sh
vivado -mode tcl -stack 2000 -source vivado_alt_synth.tcl
vivado -mode tcl -stack 2000 -source vivado_alt_synth.tcl
