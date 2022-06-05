module shift_rom
    #(parameter
        ID=1,
        DATA_WIDTH=16,
        ADDR_WIDTH=9
    )(
    input clock,
    output[DATA_WIDTH-1:0] data_out
);
    reg[ADDR_WIDTH-1:0] addr;
    (* keep = "true", ram_style = "block" *) reg[DATA_WIDTH-1:0] reg_array[2**ADDR_WIDTH-1:0];
    (* keep = "true" *) reg[DATA_WIDTH-1:0] data_out_reg;

    integer i;
    initial begin
        for (i = 0; i < 100; i = i+1)
            begin
                reg_array[i] <= 0;
            end
        addr = 0;
    end

    always @(posedge clock) begin
        if (addr == 2 ** ADDR_WIDTH) begin
            addr = 0;
        end else begin
            addr <= addr+1'b1;
        end
        data_out_reg <= reg_array[addr];
    end

    assign data_out = data_out_reg;
endmodule
