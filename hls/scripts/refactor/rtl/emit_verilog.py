import argparse
import enum

from dataclasses import dataclass

from hls.scripts.refactor.parse import parse_mlir_module


class RegOrWire(enum.Enum):
    REG = "reg"
    WIRE = "wire"


@dataclass(frozen=True)
class Val:
    reg_or_wire: RegOrWire
    id: str

    @property
    def name(self):
        return f"{self.id}_{self.reg_or_wire.value}"

    def __repr__(self):
        return self.name


def main(mac_rewritten_sched_mlir_str):
    op_id_data, func_args, returns, vals, pe_idxs = parse_mlir_module(
        mac_rewritten_sched_mlir_str
    )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    args = parser.parse_args()
    mac_rewritten_sched_mlir_str = open(args.fp).read()
    main(mac_rewritten_sched_mlir_str)
