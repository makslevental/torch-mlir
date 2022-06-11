module simple_dual_rw_ram #(
    parameter LAYER=1,
    parameter ID=1,
    parameter WIDTH = 16,                // size of each entry
    parameter SIZE = 512               // number of entries
  )(
    // write interface
    input wire wclk,                        // write clock
    input wire [$clog2(SIZE)-1:0] waddr,   // write address
    input wire [WIDTH-1:0] write_data,       // write data
    input wire write_en,                    // write enable (1 = write)
    
    // read interface
    input wire rclk,                        // read clock
    input wire [$clog2(SIZE)-1:0] raddr,   // read address
    output reg [WIDTH-1:0] read_data    // read data
  );

  (* ram_style = "block" *) reg [WIDTH-1:0] mem [SIZE-1:0];      // memory array
  
  // write clock domain
  always @(posedge wclk) begin
    if (write_en)                      // if write enable
      mem[waddr] <= write_data;        // write memory
  end
  
  // read clock domain
  always @(posedge rclk) begin
    read_data <= mem[raddr];           // read memory
  end
  
endmodule