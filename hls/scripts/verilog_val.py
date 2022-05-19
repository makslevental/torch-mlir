import numpy as np


def format_cst(cst):
    return np.random.randint(1, 100000)
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


def make_args_globals(Args):
    from mlir_ops import ArrayDecl

    args = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            if arg.input:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(
                        f"\tinput [15:0] {arg.var_name}_{'_'.join(map(str, index))}"
                    )
            elif arg.output:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(
                        f"\toutput [15:0] {arg.var_name}_{'_'.join(map(str, index))}"
                    )

    return args


FILE = open("forward.v", "w")


def VerilogForward(Args, OUTPUT_ARRAYS, forward, PES):
    # args = make_args_globals(Args)
    forward()
    for pe_idx, pe in PES.items():
        print(pe_idx, len(pe._instructions))
    #
    # print(
    #     dedent(
    #         """\
    #             `timescale 1ns/1ps
    #
    #             module forward(
    #                 output ap_local_block,
    #                 output ap_local_deadlock,
    #                 input ap_clk,
    #                 input ap_rst,
    #                 input ap_start,
    #                 output ap_done,
    #                 output ap_idle,
    #                 output ap_ready,
    #         """
    #     ),
    #     file=FILE,
    # )
    #
    # for i, inp in enumerate(args):
    #     print(inp, end="", file=FILE)
    #     if i < len(args) - 1:
    #         print(",\n", end="", file=FILE)
    #     else:
    #         print("\n", end="", file=FILE)
    #
    # print(");", file=FILE)
    #
    # forward()
    #
    # for mac in MACS.values():
    #     print(make_mac(mac.id), file=FILE)
    #
    # for relu in RELUS.values():
    #     print(make_relu(relu.id), file=FILE)
    #
    # fsm_width = 0
    # for _mac_idx, mac in MACS.items():
    #     fsm_width = max(fsm_width, int(log2(len(mac.work))) + 1)
    #
    # print(
    #     indent(
    #         dedent(
    #             f"""
    # reg [{fsm_width - 1}:0] fsm_state;
    # reg [{fsm_width - 1}:0] fsm_state_next;
    #
    # always @(*) begin
    #    fsm_state_next = fsm_state + 1;
    # end
    #
    # always @(posedge ap_clk or negedge ap_rst) begin
    #    if (!ap_rst)
    #       fsm_state <= {fsm_width}'b0;
    #    else
    #       fsm_state <= fsm_state_next;
    # end
    # """
    #         ),
    #         "\t",
    #     ),
    #     file=FILE,
    # )
    #
    # for mac_idx, mac in MACS.items():
    #     print(indent(f"// mac idx {mac_idx}", "\t"), file=FILE)
    #     print(indent("always @ (fsm_state) begin", "\t"), file=FILE)
    #     print(indent("case(fsm_state)", "\t\t"), file=FILE)
    #     for i, assigns in enumerate(mac.work):
    #         if isinstance(assigns, tuple):
    #             print(indent(f"{fsm_width}'d{i} : begin", "\t\t\t"), file=FILE)
    #             for assign in assigns:
    #                 print(
    #                     indent(f'{assign.replace("<=", "=")};', "\t\t\t\t"), file=FILE
    #                 )
    #             print(indent("end", "\t\t\t"), file=FILE)
    #         else:
    #             print(
    #                 indent(f"{fsm_width}'d{i} : {assigns};", "\t\t\t"),
    #                 file=FILE,
    #             )
    #     print(indent("endcase", "\t\t"), file=FILE)
    #     print(indent("end", "\t"), file=FILE)
    #     print(file=FILE)
    #
    # # while any_rounds_left():
    # #     round = []
    # #     for mac_idx, macs in MAC_STACKS.items():
    # #         round.append(macs.popleft())
    # #     rounds.append(round)
    # #
    # # for i, round in enumerate(rounds):
    # #     if not len(round) or isinstance(round[0], MAC_TERMINAL):
    # #         # TODO end accum
    # #         continue
    # #
    # #     print(indent("always @(*) begin", "\t"), file=FILE)
    # #     for assigns in round:
    # #         for assign in assigns:
    # #             print(indent(f"{assign};", "\t\t"), file=FILE)
    # #     print(indent("end", "\t"), file=FILE)
    # #     print(file=FILE)
    #
    # for arr in OUTPUT_ARRAYS:
    #     for idx, reg in arr.registers.items():
    #         out_wire = "_arg1_" + "_".join(map(str, idx))
    #         print(indent(f"assign {out_wire} = {reg};", "\t"), file=FILE)
    #     print(file=FILE)
    #
    # print("endmodule", file=FILE)


def test_hex():
    print(format_cst(1000 + np.random.randn(1)[0]))
