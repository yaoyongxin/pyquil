##############################################################################
# Copyright 2016-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""
Module for parsing Quil programs from text into PyQuil objects
"""
import operator

from lark import Lark, Transformer, v_args
import numpy as np
from lark.exceptions import LarkError

from pyquil.quil import Program

from pyquil._parser.PyQuilListener import run_parser
from pyquil.quilatom import Qubit, MemoryReference, Addr, Label
from pyquil.quilbase import Gate, DefGate, Measurement, JumpTarget, Halt, Jump, JumpWhen, JumpUnless, Reset, ResetQubit, \
    Wait, Nop, RawInstr, Pragma


def parse_program_old(quil):
    """
    Parse a raw Quil program and return a PyQuil program.

    :param str quil: a single or multiline Quil program
    :return: PyQuil Program object
    """
    return Program(parse_old(quil))


def parse_old(quil):
    """
    Parse a raw Quil program and return a corresponding list of PyQuil objects.

    :param str quil: a single or multiline Quil program
    :return: list of instructions
    """
    return run_parser(quil)


quil_grammar = """
// Top level
quil                : all_instr*
?all_instr          : instr
                    | def_gate
?instr              : gate
                    | measure
                    | control
                    | reset
                    | wait
                    | nop
                    | include
                    | pragma

// Section C. Static and Parametric Gates
gate                : name [ "(" param ( "," param )* ")" ] qubit+
?param              : expression
qubit               : int_n

// Section D. Gate Definitions
def_gate            : "DEFGATE" name ( "(" variable ( "," variable )* ")" )? ":" matrix

variable            : "%" name
matrix              : ( _NEWLINE_TAB matrix_row )*
matrix_row          : expression ( "," expression )*

// E. Circuits

// F. Measurement

measure             : "MEASURE" qubit addr?
?addr               : name [ "[" int_n "]" ] -> memory_ref
                    | int_n -> classical_addr
                    
// G. Program control

?control            : "LABEL" label -> def_label
                    | "HALT" -> halt
                    | "JUMP" label -> jump
                    | "JUMP-WHEN" label addr -> jump_when
                    | "JUMP-UNLESS" label addr -> jump_unless
label               : "@" name

// H. Zeroing the Quantum State

reset               : "RESET" qubit?

// I. Classical/Quantum Synchronization

wait                : "WAIT"

// J. Classical Instructions

// K. The No-Operation Instruction

nop                 : "NOP"

// L. File Inclusion

include             : "INCLUDE" string

// M. Pragma Support

pragma              : "PRAGMA" name pragma_name* string?
?pragma_name        : name | int_n

// Expressions

?expression         : product
                    | expression "+" product -> add
                    | expression "-" product -> sub

?product            : power
                    | product "*" power -> mul
                    | product "/" power -> div

?power              : atom
                    | atom "^" power -> pow

?atom               : number
                    | "-" atom -> neg
                    | "+" atom -> pos
                    | FUNCTION "(" expression ")" -> apply_fun
                    | "(" expression ")"

FUNCTION            : "sin" | "cos" | "sqrt" | "exp" | "cis"

// Numbers

?number             : (int_n|float_n) "i" -> imag
                    | int_n
                    | float_n
                    | "i" -> i
                    | "pi" -> pi
                    
int_n               : INT
float_n             : FLOAT

// Common

name                : IDENTIFIER
IDENTIFIER          : ("_"|LETTER) [ ("_"|"-"|LETTER|DIGIT)* ("_"|LETTER|DIGIT) ]

string              : ESCAPED_STRING

_NEWLINE_TAB        : NEWLINE "    "

%import common.DIGIT
%import common.ESCAPED_STRING
%import common.FLOAT
%import common.INT
%import common.LETTER
%import common.NEWLINE
%import common.WS

%ignore WS
"""


@v_args(inline=True)
class QuilTransformer(Transformer):
    def quil(self, *instructions):
        return list(instructions)

    def gate(self, name, *args):
        params = [arg for arg in args if not isinstance(arg, Qubit)]
        qubits = [arg for arg in args if isinstance(arg, Qubit)]
        return Gate(name, params=params, qubits=qubits)

    qubit = Qubit

    def def_gate(self, name, *args):
        *variables, matrix = args
        return DefGate(name, matrix=np.array(matrix), parameters=variables)

    def matrix(self, *rows):
        return list(rows)

    def matrix_row(self, *expressions):
        return list(expressions)

    def measure(self, qubit, *addr):
        return Measurement(qubit, addr[0] if addr else None)

    def memory_ref(self, name, *n):
        return MemoryReference(name, n[0] if n else 0)

    def classical_addr(self, n):
        return Addr(n)

    def def_label(self, label):
        return JumpTarget(label)

    def halt(self):
        return Halt()

    def jump(self, label):
        return Jump(label)

    def jump_when(self, label, addr):
        return JumpWhen(label, addr)

    def jump_unless(self, label, addr):
        return JumpUnless(label, addr)

    def label(self, name):
        return Label(name)

    def reset(self, *qubit):
        if qubit:
            return ResetQubit(qubit[0])
        else:
            return Reset()

    def wait(self):
        return Wait()

    def nop(self):
        return Nop()

    def include(self, string):
        return RawInstr(string)

    def pragma(self, name, *args):
        if args and isinstance(args[-1], str) and args[-1][0] == '"':
            freeform_string = args[-1][1:-1]
            args = args[:-1]
        else:
            freeform_string = ""
        return Pragma(name, args=args, freeform_string=freeform_string)

    add = operator.add
    sub = operator.sub
    mul = operator.mul
    div = operator.truediv
    pow = operator.pow
    neg = operator.neg
    pos = operator.pos

    def apply_fun(self, fun, arg):
        if fun == "sin":
            return np.sin(arg)
        if fun == "cos":
            return np.cos(arg)
        if fun == "sqrt":
            return np.sqrt(arg)
        if fun == "exp":
            return np.exp(arg)
        if fun == "cis":
            return np.cos(arg) + 1j * np.sin(arg)

    def imag(self, n):
        return n * 1j

    int_n = int
    float_n = float

    def i(self):
        return 1j

    def pi(self):
        return np.pi

    name = str
    string = str


quil_parser = Lark(quil_grammar, start='quil', parser='lalr', transformer=QuilTransformer())


def parse_program(quil) -> Program:
    """
    Parse a raw Quil program and return a PyQuil program.

    :param str quil: a single or multiline Quil program
    :return: PyQuil Program object
    """
    return Program(parse(quil))


def parse(quil):
    """
    Parse a raw Quil program and return a corresponding list of PyQuil objects.

    :param str quil: a single or multiline Quil program
    :return: list of instructions
    """
    try:
        return quil_parser.parse(quil)
    except LarkError as e:
        raise RuntimeError(e)
