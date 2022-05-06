// • It is the first summand after aresetn has been asserted and released.
module mac(
    input wire aclk,
    input wire aresetn,
    input wire[15:0] a_tdata,
    input wire a_tlast,
    input wire[15:0] b_tdata,
    output wire[15:0] r_tdata,
    output wire r_tlast
);
    reg[15:0] mul_reg;
    wire[15:0] mul_wire;
    mul_half_precision local_mul_half_precision(
        .aclk(aclk),
        .s_axis_a_tvalid(1'b1),
        .s_axis_a_tdata(a_tdata),
        .s_axis_b_tvalid(1'b1),
        .s_axis_b_tdata(b_tdata),
        .m_axis_result_tvalid(1'b1),
        .m_axis_result_tdata(mul_wire)
    );

    accum_half_precision local_accum_half_precision(
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_a_tvalid(1'b1),
        .s_axis_a_tdata(mul_reg),
        .s_axis_a_tlast(a_tlast),
        .m_axis_result_tvalid(1'b1),
        .m_axis_result_tdata(r_tdata),
        .m_axis_result_tlast(r_tlast)
    );

    always @(posedge aclk) begin
        mul_reg <= mul_wire;
    end

endmodule
