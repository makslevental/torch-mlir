`timescale 1ns/1ps
module relu
    #(parameter
        ID=1,
        WIDTH=16
    )(
    input[WIDTH-1:0] a,
    output[WIDTH-1:0] z
);
    assign z = (a[WIDTH-1] == 0) ? a:0;   //if the sign bit is high, send zero on the output else send the input
endmodule
