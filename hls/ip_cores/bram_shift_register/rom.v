module rom
    #(parameter
        ID=1,
        DATA_WIDTH=16,
        ADDR_WIDTH=9
    )(
    input clock,
    output[DATA_WIDTH-1:0] data_out
);

    reg[ADDR_WIDTH-1:0] addr;
    (* ram_style = "block" *) reg[DATA_WIDTH-1:0] mem[2**ADDR_WIDTH-1:0];
     output_reg = 8'd0;

    initial begin
        mem[4'd0] = 8'h00;
        mem[4'd1] = 8'h00;
        mem[4'd2] = 8'h00;
        mem[4'd3] = 8'h00;
        mem[4'd4] = 8'h00;
        mem[4'd5] = 8'h00;
        mem[4'd6] = 8'h00;
        mem[4'd7] = 8'h00;
        mem[4'd8] = 8'h00;
        mem[4'd9] = 8'h00;
        mem[4'd10] = 8'h00;
        mem[4'd11] = 8'h00;
        mem[4'd12] = 8'h00;
        mem[4'd13] = 8'h00;
        mem[4'd14] = 8'h00;
        mem[4'd15] = 8'h00;
        mem[4'd16] = 8'h00;
    end

    assign data = output_reg;

    always @(posedge clock) begin
        output_reg <= mem[addr];
    end

endmodule