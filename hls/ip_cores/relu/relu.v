`timescale 1ns / 1ps
module relu(
    input [31:0] din_relu,
    output [31:0] dout_relu
  );
  assign dout_relu = (din_relu[31] == 0)? din_relu : 0;   //if the sign bit is high, send zero on the output else send the input
endmodule
