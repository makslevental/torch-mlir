import math
from textwrap import dedent, indent


def get_fsm_idx_width(max_fsm_stage):
    return math.ceil(math.log10(max_fsm_stage))


class FSM:
    def __init__(self, max_fanout, max_fsm_stage):
        self.max_fanout = max_fanout
        self.max_fsm_stage = max_fsm_stage
        self.fsm_idx_width = math.ceil(math.log10(max_fsm_stage))

    def make_fsm_states(self, fsm_states):
        s = " | ".join(
            [
                f"(1'b1 == current_state_fsm_state{str(i).zfill(self.fsm_idx_width)})"
                for i in fsm_states
            ]
        )
        return s

    def make_fsm_params(self):
        params = "\n".join(
            [
                f"parameter fsm_state{str(i).zfill(self.fsm_idx_width)} = {self.max_fsm_stage + 1}'d{1 << i - 1};"
                for i in range(1, self.max_fsm_stage + 1)
            ]
        )
        params += "\n\n"
        params += f'(* max_fanout = {self.max_fanout}, fsm_encoding = "none" *) reg [{self.max_fsm_stage}:0] current_state_fsm;\n'
        params += f"reg [{self.max_fsm_stage}:0] next_state_fsm;"

        return params

    def make_fsm_wires(self):
        wires = "\n".join(
            [
                f"wire current_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(1, self.max_fsm_stage + 1)
            ]
        )
        return wires

    def make_fsm(self):
        first_state = str(1).zfill(self.fsm_idx_width)
        fsm = dedent(
            f"""\
                always @ (posedge clk) begin
                    if (reset == 1'b1) begin
                        current_state_fsm <= fsm_state{first_state};
                    end else begin
                        current_state_fsm <= next_state_fsm;
                    end
                end

                always @ (*) begin
                    case (current_state_fsm)
                """
        )
        for i in range(1, self.max_fsm_stage):
            fsm += indent(
                dedent(
                    f"""\
                    fsm_state{str(i).zfill(self.fsm_idx_width)} : begin
                        next_state_fsm = fsm_state{str(i + 1).zfill(self.fsm_idx_width)};
                    end
                    """
                ),
                "\t\t",
            )

        fsm += indent(
            dedent(
                f"""\
                fsm_state{str(self.max_fsm_stage).zfill(self.fsm_idx_width)} : begin
                    next_state_fsm = fsm_state{str(1).zfill(self.fsm_idx_width)};
                end
                """
            ),
            "\t\t",
        )

        fsm += indent(
            dedent(
                """\
        default : begin
            next_state_fsm = 'bx;
        end
        """
            ),
            "\t\t",
        )
        fsm += dedent(
            """\
            endcase
        end
        """
        )

        fsm_2bit_width = math.ceil(math.log2(self.max_fsm_stage))
        for i in range(1, self.max_fsm_stage + 1):
            fsm += dedent(
                f"""\
            assign current_state_fsm_state{str(i).zfill(self.fsm_idx_width)} = current_state_fsm[{fsm_2bit_width}'d{i - 1}];
                """
            )

        return fsm
