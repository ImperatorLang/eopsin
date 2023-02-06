import ast

from enum import Enum, auto

from .typed_ast import *

import pluthon as plt
import uplc.ast as uplc


class PythonBuiltIn(Enum):
    all = plt.Lambda(
        ["xs", "_"],
        plt.FoldList(
            plt.Var("xs"),
            plt.Lambda(["x", "a"], plt.And(plt.Var("x"), plt.Var("a"))),
            plt.Integer(0),
        ),
    )
    any = plt.Lambda(
        ["xs", "_"],
        plt.FoldList(
            plt.Var("xs"),
            plt.Lambda(["x", "a"], plt.Or(plt.Var("x"), plt.Var("a"))),
            plt.Integer(0),
        ),
    )
    abs = plt.Lambda(
        ["x", "_"],
        plt.Ite(
            plt.LessThanInteger(plt.Var("x"), plt.Integer(0)),
            plt.SubtractInteger(plt.Integer(0), plt.Var("x")),
            plt.Var("x"),
        ),
    )
    breakpoint = plt.Lambda(["_"], plt.NoneData())
    len = auto()
    print = plt.Lambda(
        ["x", "_"],
        plt.Trace(plt.Var("x"), plt.NoneData()),
    )
    range = plt.Lambda(
        ["limit", "_"],
        plt.Range(plt.Var("limit")),
    )
    sum = plt.Lambda(
        ["xs", "_"],
        plt.FoldList(
            plt.Var("xs"), plt.BuiltIn(uplc.BuiltInFun.AddInteger), plt.Integer(0)
        ),
    )


class Len(PolymorphicFunction):
    def type_from_args(self, args: typing.List[Type]) -> FunctionType:
        assert (
            len(args) == 1
        ), f"'len' takes only one argument, but {len(args)} were given"
        assert isinstance(
            args[0], InstanceType
        ), "Can only determine length of instances"
        return FunctionType(args, IntegerInstanceType)

    def impl_from_args(self, args: typing.List[Type]) -> plt.AST:
        arg = args[0]
        assert isinstance(arg, InstanceType), "Can only determine length of instances"
        if arg == ByteStringInstanceType:
            return plt.Lambda(["x", "_"], plt.LengthOfByteString(plt.Var("x")))
        elif isinstance(arg.typ, ListType):
            # simple list length function
            return plt.Lambda(
                ["x", "_"],
                plt.FoldList(
                    plt.Var("x"),
                    plt.Lambda(
                        ["a", "_"], plt.AddInteger(plt.Var("a"), plt.Integer(1))
                    ),
                    plt.Integer(0),
                ),
            )
        raise NotImplementedError(f"'len' is not implemented for type {arg}")


PythonBuiltInTypes = {
    PythonBuiltIn.all: InstanceType(
        FunctionType(
            [InstanceType(ListType(BoolInstanceType))],
            BoolInstanceType,
        )
    ),
    PythonBuiltIn.any: InstanceType(
        FunctionType(
            [InstanceType(ListType(BoolInstanceType))],
            BoolInstanceType,
        )
    ),
    PythonBuiltIn.abs: InstanceType(
        FunctionType(
            [IntegerInstanceType],
            IntegerInstanceType,
        )
    ),
    PythonBuiltIn.breakpoint: InstanceType(FunctionType([], NoneInstanceType)),
    PythonBuiltIn.len: InstanceType(PolymorphicFunctionType(Len())),
    PythonBuiltIn.print: InstanceType(
        FunctionType([StringInstanceType], NoneInstanceType)
    ),
    PythonBuiltIn.range: InstanceType(
        FunctionType(
            [IntegerInstanceType],
            InstanceType(ListType(IntegerInstanceType)),
        )
    ),
    PythonBuiltIn.sum: InstanceType(
        FunctionType(
            [InstanceType(ListType(IntegerInstanceType))],
            IntegerInstanceType,
        )
    ),
}


class CompilerError(Exception):
    def __init__(self, orig_err: Exception, node: ast.AST, compilation_step: str):
        self.orig_err = orig_err
        self.node = node
        self.compilation_step = compilation_step


class CompilingNodeTransformer(TypedNodeTransformer):
    step = "Node transformation"

    def visit(self, node):
        try:
            return super().visit(node)
        except Exception as e:
            if isinstance(e, CompilerError):
                raise e
            raise CompilerError(e, node, self.step)


class CompilingNodeVisitor(TypedNodeVisitor):
    step = "Node visiting"

    def visit(self, node):
        try:
            return super().visit(node)
        except Exception as e:
            if isinstance(e, CompilerError):
                raise e
            raise CompilerError(e, node, self.step)


def data_from_json(j: typing.Dict[str, typing.Any]) -> uplc.PlutusData:
    if "bytes" in j:
        return uplc.PlutusByteString(bytes.fromhex(j["bytes"]))
    if "int" in j:
        return uplc.PlutusInteger(int(j["int"]))
    if "list" in j:
        return uplc.PlutusList(list(map(data_from_json, j["list"])))
    if "map" in j:
        return uplc.PlutusMap({d["k"]: d["v"] for d in j["map"]})
    if "constructor" in j and "fields" in j:
        return uplc.PlutusConstr(j["constructor"], j["fields"])
    raise NotImplementedError(f"Unknown datum representation {j}")
