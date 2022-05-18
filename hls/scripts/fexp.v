//
// Copyright 2021 Xilinx, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// This module describes SIMD Inference
// 4 adders packed into single DSP block
// `timescale 100ps/100ps

(* use_dsp = "simd" *)
(* dont_touch = "1" *)
module fexp_XXX(input ap_clk, ap_rst, ap_ce, ap_start, ap_continue,
    input[31:0] a1,
    output ap_idle, ap_done, ap_ready,
    output z1_ap_vld,
    output reg[31:0] z1);

    wire ce = ap_ce;

    reg[31:0] areg1;
    reg dly1, dly2;

    always @(posedge ap_clk)
        if (ap_rst)
            begin
                z1 <= 0;
                areg1 <= 0;
                dly1 <= 0;
                dly2 <= 0;
            end
        else if (ce)
            begin
                z1 <= areg1+32'd0;
                areg1 <= a1;
                dly1 <= ap_start;
                dly2 <= dly1;
            end

    assign z1_ap_vld = dly2;
    assign ap_ready = dly2;
    assign ap_done = dly2;
    assign ap_idle = ~ap_start;

endmodule // rtl_model
