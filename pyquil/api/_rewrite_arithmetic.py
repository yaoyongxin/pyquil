from numbers import Real
from typing import Dict, Tuple

from pyquil import Program
from pyquil.quilatom import MemoryReference, Expression
from pyquil.quilbase import (
    Declare,
    Gate,
)
from rpcq.messages import (
    ParameterSpec,
    ParameterAref,
    RewriteArithmeticResponse
)

def rewrite_arithmetic(prog: Program) -> RewriteArithmeticResponse:
    """ Rewrite compound arithmetic expressions.
    """
    def spec(inst: Declare) -> ParameterSpec:
        return ParameterSpec(type=inst.memory_type, length=inst.memory_size)

    def aref(ref: MemoryReference) -> ParameterAref:
        return ParameterAref(name=ref.name, index=ref.offset)

    updated = prog.copy_everything_except_instructions()
    old_descriptors = {
        inst.name: spec(inst) for inst in prog if isinstance(inst, Declare)
    }
    recalculation_table = {}
    seen_exprs = set()

    mref_name = "__P" + str(len(old_descriptors))
    mref_idx = 0

    for inst in prog:
        if isinstance(inst, Gate):
            new_params = []
            for param in inst.params:
                if isinstance(param, (Real, MemoryReference)):
                    new_params.append(param)
                elif isinstance(param, Expression):
                    expr = str(param)
                    if expr in seen_exprs:
                        new_params.append(param)
                    else:
                        new_mref = MemoryReference(mref_name, mref_idx)
                        mref_idx += 1
                        seen_exprs.add(expr)
                        recalculation_table[aref(new_mref)] = expr
                        new_params.append(new_mref)
                else:
                    raise ValueError(f"Unknown parameter type {type(param)} in {inst}.")
            updated.inst(Gate(inst.name, new_params, inst.qubits))
        # TODO: consider how to handle frame mutations
        else:
            updated.inst(inst)

    if mref_idx > 0:
        updated._instructions.insert(0, Declare(mref_name, 'REAL', mref_idx))

    return RewriteArithmeticResponse(
        quil=updated.out(),
        original_memory_descriptors=old_descriptors,
        recalculation_table=recalculation_table
    )
