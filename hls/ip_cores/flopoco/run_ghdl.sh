docker run --rm=true -v `pwd`:/flopoco_workspace flopoco FPAdd we=4 wf=4 outputFile=sfp_add.vhdl target=Virtex6 frequency=800 name=forward pipeline=1 useHardMult=1
ghdl -a --ieee=synopsys -fexplicit sfp_add.vhdl
ghdl -a --ieee=synopsys -fexplicit  full_adder_testbench.vhdl
ghdl -e --ieee=synopsys -fexplicit full_adder_testbench
ghdl -r --ieee=synopsys -fexplicit full_adder_testbench --vcd=add.vcd

docker run --rm=true -v `pwd`:/flopoco_workspace flopoco FPMult we=4 wf=4 outputFile=sfp_mult.vhdl target=Virtex6 frequency=800 name=forward pipeline=1 useHardMult=1
ghdl -a --ieee=synopsys -fexplicit sfp_mult.vhdl
ghdl -a --ieee=synopsys -fexplicit  full_adder_testbench.vhdl
ghdl -e --ieee=synopsys -fexplicit full_adder_testbench
ghdl -r --ieee=synopsys -fexplicit full_adder_testbench --vcd=mult.vcd

