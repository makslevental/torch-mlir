import random
from collections import defaultdict
from textwrap import dedent

from dataclasses import dataclass


@dataclass(frozen=True)
class Wire:
    id: str
    precision: int

    def __str__(self):
        return f"{self.id}_wire"

    def instantiate(self):
        return f"wire [{self.precision - 1}:0] {self};"


@dataclass(frozen=True)
class Reg:
    id: str
    precision: int

    def __str__(self):
        return f"{self.id}_reg"

    def instantiate(self):
        return f"reg [{self.precision - 1}:0] {self};"


def make_constant(v, precision):
    return f"{precision}'d{random.randint(0, 2 ** precision - 1)}"


def make_always_tree(left, right, cond):
    return dedent(
        f"""\
        always @ (*) begin
            if ({cond}) begin
                {left} = {right}; 
        end"""
    )
