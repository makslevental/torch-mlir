module fma(
    input wire aclk,
    input wire[15:0] a_tdata,
    input wire[15:0] b_tdata,
    input wire[15:0] c_tdata,
    output wire[15:0] r_tdata
);
    fma_half_precision here_fma_half_precision(
        .aclk(aclk),
        .s_axis_a_tvalid(1'b1),
        .s_axis_a_tdata(a_tdata),
        .s_axis_b_tvalid(1'b1),
        .s_axis_b_tdata(b_tdata),
        .s_axis_c_tvalid(1'b1),
        .s_axis_c_tdata(c_tdata),
        .m_axis_result_tvalid(1'b1),
        .m_axis_result_tdata(r_tdata)
    );
endmodule
