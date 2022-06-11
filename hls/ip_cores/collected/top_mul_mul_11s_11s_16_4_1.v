
`timescale 1 ns / 1 ps

  module top_mul_mul_11s_11s_16_4_1_DSP48_0(clk, rst, ce, a, b, p);
input wire clk;
input wire rst;
input wire ce;
input signed [11 - 1 : 0] a;
input signed [11 - 1 : 0] b;
output signed [16 - 1 : 0] p;

reg signed [16 - 1 : 0] p_reg; 

reg signed [11 - 1 : 0] a_reg; 
reg signed [11 - 1 : 0] b_reg; 

reg signed [16 - 1 : 0] p_reg_tmp; 

always @ (posedge clk) begin
    if (ce) begin
        a_reg <= a;
        b_reg <= b;
        p_reg_tmp <= a_reg * b_reg;
        p_reg <= p_reg_tmp;
    end
end

assign p = p_reg;

endmodule
`timescale 1 ns / 1 ps
module fmul
    #(parameter
        LAYER=1,
        ID=1,
        WIDTH=16
    )(
    input wire clock,
    input wire reset,
    input wire clock_enable,
    input wire[WIDTH-1:0] a,
    input wire[WIDTH-1:0] b,
    output wire[WIDTH-1:0] res
);;



top_mul_mul_11s_11s_16_4_1_DSP48_0 top_mul_mul_11s_11s_16_4_1_DSP48_0_U(
    .clk( clock ),
    .rst( reset ),
    .ce( clock_enable ),
    .a( a ),
    .b( b ),
    .p( res ));

endmodule

