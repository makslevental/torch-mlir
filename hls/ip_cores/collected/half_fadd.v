// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.3.0 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

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
//------------------------Local signal-------------------
    wire aclock;
    wire aclocken;
    wire a_tvalid;
    wire[15:0] a_tdata;
    wire b_tvalid;
    wire[15:0] b_tdata;
    wire r_tvalid;
    wire[15:0] r_tdata;
    reg clock_enable_r;
    wire[WIDTH-1:0] dout_i;
    reg[WIDTH-1:0] dout_r;
//------------------------Instantiation------------------
    half_fadd _half_fadd(
        .aclk(aclock),
        .s_axis_a_tvalid(a_tvalid),
        .s_axis_a_tdata(a_tdata),
        .s_axis_b_tvalid(b_tvalid),
        .s_axis_b_tdata(b_tdata),
        //.s_axis_operation_tdata({1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0}),
        //.s_axis_operation_tvalid(1'b0),
        .m_axis_result_tvalid(r_tvalid),
        .m_axis_result_tdata(r_tdata)
    );
//------------------------Body---------------------------
    assign aclock = clock;
    assign aclocken = clock_enable_r;
    assign a_tvalid = 1'b1;
    assign a_tdata = a;
    assign b_tvalid = 1'b1;
    assign b_tdata = b;
    assign dout_i = r_tdata;

    always @(posedge clock) begin
        clock_enable_r <= clock_enable;
    end

    always @(posedge clock) begin
        if (clock_enable_r) begin
            dout_r <= dout_i;
        end
    end

    assign res = clock_enable_r ? dout_i : dout_r;
endmodule
