// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2021.2 (64-bit)
// Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps

module top_add_11ns_11ns_11_2_1_Adder_0(clk, reset, ce, a, b, s);

// ---- input/output ports list here ----
input wire  clk;
input wire  reset;
input wire  ce;
input  wire [11 - 1 : 0] a;
input  wire [11 - 1 : 0] b;
output wire [11 - 1 : 0] s;

// wire for the primary inputs
wire [11 - 1 : 0] ain_s0 = a;
wire [11 - 1 : 0] bin_s0 = b;

// This AddSub module have totally 2 stages. For each stage the adder's width are:
// 5 6

// Stage 1 logic
wire [5 - 1 : 0]     fas_s1;
wire                 facout_s1;
reg  [6 - 1 : 0]     ain_s1;
reg  [6 - 1 : 0]     bin_s1;
reg  [5 - 1 : 0]     sum_s1;
reg                  carry_s1;
top_add_11ns_11ns_11_2_1_Adder_0_comb_adder #(
    .N    ( 5 )
) u1 (
    .a    ( ain_s0[5 - 1 : 0] ),
    .b    ( bin_s0[5 - 1 : 0] ),
    .cin  ( 1'b0 ),
    .s    ( fas_s1 ),
    .cout ( facout_s1 )
);

always @ (posedge clk) begin
    if (ce) begin
        sum_s1   <= fas_s1;
        carry_s1 <= facout_s1;
    end
end

always @ (posedge clk) begin
    if (ce) begin
        ain_s1 <= ain_s0[11 - 1 : 5];
    end
end

always @ (posedge clk) begin
    if (ce) begin
        bin_s1 <= bin_s0[11 - 1 : 5];
    end
end

// Stage 2 logic
wire [6 - 1 : 0]     fas_s2;
wire                 facout_s2;
top_add_11ns_11ns_11_2_1_Adder_0_comb_adder #(
    .N    ( 6 )
) u2 (
    .a    ( ain_s1[6 - 1 : 0] ),
    .b    ( bin_s1[6 - 1 : 0] ),
    .cin  ( carry_s1 ),
    .s    ( fas_s2 ),
    .cout ( facout_s2 )
);

assign s = {fas_s2, sum_s1};

endmodule

// small adder
module top_add_11ns_11ns_11_2_1_Adder_0_comb_adder 
#(parameter
    N = 32
)(
    input  wire [N-1 : 0]  a,
    input  wire [N-1 : 0]  b,
    input  wire           cin,
    output wire [N-1 : 0]  s,
    output wire           cout
);
assign {cout, s} = a + b + cin;

endmodule

`timescale 1 ns / 1 ps
module fadd
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
);

top_add_11ns_11ns_11_2_1_Adder_0 top_add_11ns_11ns_11_2_1_Adder_0_U(
    .clk( clock ),
    .reset( reset ),
    .ce( clock_enable ),
    .a( a ),
    .b( b ),
    .s( res ));

endmodule

