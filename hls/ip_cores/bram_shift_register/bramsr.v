module bramsr(
    input ap_clk,             //system clock
    input reset_n,         //asynchronous active low reset
    input[7:0] shift_in,    //two input signals to be debounced
    output[7:0] shift_out    //two debounced signals
);

    srl_514_bram#(514, 9, 8)
        bram_shifter(
            .clk(ap_clk),
            .shift_in(shift_in),
            .shift_out(shift_out)
        );

endmodule