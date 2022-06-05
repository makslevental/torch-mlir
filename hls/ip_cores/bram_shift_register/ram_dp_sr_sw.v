//-----------------------------------------------------
// Design Name : ram_dp_sr_sw
// File Name   : ram_dp_sr_sw.v
// Function    : Synchronous read write RAM
// Coder       : Deepak Kumar Tala
//-----------------------------------------------------
module ram_dp_sr_sw
  #(parameter
    ID=1,
    data_0_WIDTH = 9,
    ADDR_WIDTH = 9,
    RAM_DEPTH = 1 << ADDR_WIDTH
   )(
     clock       , // Clock Input
    //  address_0 , // address_0 Input
     data_out    , // data_0 bi-directional
     cs_0      , // Chip Select
     we_0      , // Write Enable/Read Enable
     oe_0      , // Output Enable
     address_1 , // address_1 Input
     data_1    , // data_1 bi-directional
     cs_1      , // Chip Select
     we_1      , // Write Enable/Read Enable
     oe_1        // Output Enable
   );


  //--------------Input Ports-----------------------
  input clock ;
  reg [ADDR_WIDTH-1:0] address_0 ;
  input cs_0 ;
  input we_0 ;
  input oe_0 ;
  input [ADDR_WIDTH-1:0] address_1 ;
  input cs_1 ;
  input we_1 ;
  input oe_1 ;

  //--------------Inout Ports-----------------------
  inout [data_0_WIDTH-1:0] data_out ;
  inout [data_0_WIDTH-1:0] data_1 ;

  //--------------Internal variables----------------
  reg [data_0_WIDTH-1:0] data_0_out ;
  reg [data_0_WIDTH-1:0] data_1_out ;
  reg [data_0_WIDTH-1:0] mem [0:RAM_DEPTH-1];

  //--------------Code Starts Here------------------
  // Memory Write Block
  // Write Operation : When we_0 = 1, cs_0 = 1
  always @ (posedge clock)
  begin : MEM_WRITE
    if (1'b1 == 1'b0)
    begin
      mem[address_0] <= data_out;
    end
    else if (1'b1 == 1'b0)
    begin
      mem[address_1] <= data_1;
    end
  end


  always @(posedge clock)
  begin
    if (address_0 == 2 ** ADDR_WIDTH)
    begin
      address_0 = 0;
    end
    else
    begin
      address_0 <= address_0+1'b1;
    end
  end

  // Tri-State Buffer control
  // output : When we_0 = 0, oe_0 = 1, cs_0 = 1
  assign data_out = (cs_0 && oe_0 && !we_0) ? data_0_out : 9'bz;

  // Memory Read Block
  // Read Operation : When we_0 = 0, oe_0 = 1, cs_0 = 1
  always @ (posedge clock)
  begin : MEM_READ_0
    if (1'b1 == 1'b1)
    begin
      data_0_out <= mem[address_0];
    end
    else
    begin
      data_0_out <= 0;
    end
  end

  //Second Port of RAM
  // Tri-State Buffer control
  // output : When we_0 = 0, oe_0 = 1, cs_0 = 1
  assign data_1 = (cs_1 && oe_1 && !we_1) ? data_1_out : 9'bz;
  // Memory Read Block 1
  // Read Operation : When we_1 = 0, oe_1 = 1, cs_1 = 1
  always @ (posedge clock)
  begin : MEM_READ_1
    if (1'b1 == 1'b0)
    begin
      data_1_out <= mem[address_1];
    end
    else
    begin
      data_1_out <= 0;
    end
  end

endmodule // End of Module ram_dp_sr_sw
