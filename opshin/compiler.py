import logging
from pycardano import PlutusData

from uplc.ast import data_from_cbor
from .optimize.optimize_const_folding import OptimizeConstantFolding
from .optimize.optimize_inline_constvars import OptimizeInlineConstvars
from .optimize.optimize_remove_comments import OptimizeRemoveDeadconstants
from .rewrite.rewrite_augassign import RewriteAugAssign
from .rewrite.rewrite_cast_condition import RewriteConditions
from .rewrite.rewrite_comparison_chaining import RewriteComparisonChaining
from .rewrite.rewrite_forbidden_overwrites import RewriteForbiddenOverwrites
from .rewrite.rewrite_import import RewriteImport
from .rewrite.rewrite_import_dataclasses import RewriteImportDataclasses
from .rewrite.rewrite_import_hashlib import RewriteImportHashlib
from .rewrite.rewrite_import_integrity_check import RewriteImportIntegrityCheck
from .rewrite.rewrite_import_plutusdata import RewriteImportPlutusData
from .rewrite.rewrite_import_typing import RewriteImportTyping
from .rewrite.rewrite_import_uplc_builtins import RewriteImportUPLCBuiltins
from .rewrite.rewrite_inject_builtins import RewriteInjectBuiltins
from .rewrite.rewrite_inject_builtin_constr import RewriteInjectBuiltinsConstr
from .rewrite.rewrite_orig_name import RewriteOrigName
from .rewrite.rewrite_remove_type_stuff import RewriteRemoveTypeStuff
from .rewrite.rewrite_scoping import RewriteScoping
from .rewrite.rewrite_subscript38 import RewriteSubscript38
from .rewrite.rewrite_tuple_assign import RewriteTupleAssign
from .optimize.optimize_remove_pass import OptimizeRemovePass
from .optimize.optimize_remove_deadvars import OptimizeRemoveDeadvars
from .optimize.optimize_varlen import OptimizeVarlen
from .type_inference import *
from .util import (
    CompilingNodeTransformer,
    NoOp,
)
from .typed_ast import (
    transform_ext_params_map,
    transform_output_map,
    RawPlutoExpr,
    PowImpl,
    ByteStrIntMulImpl,
    StrIntMulImpl,
)

_LOGGER = logging.getLogger(__name__)

STATEMONAD = "s"


BinOpMap = {
    Add: {
        IntegerInstanceType: {
            IntegerInstanceType: plt.AddInteger,
        },
        ByteStringInstanceType: {
            ByteStringInstanceType: plt.AppendByteString,
        },
        StringInstanceType: {
            StringInstanceType: plt.AppendString,
        },
    },
    Sub: {
        IntegerInstanceType: {
            IntegerInstanceType: plt.SubtractInteger,
        }
    },
    Mult: {
        IntegerInstanceType: {
            IntegerInstanceType: plt.MultiplyInteger,
            ByteStringInstanceType: lambda x, y: ByteStrIntMulImpl(y, x),
            StringInstanceType: lambda x, y: StrIntMulImpl(y, x),
        },
        StringInstanceType: {
            IntegerInstanceType: StrIntMulImpl,
        },
        ByteStringInstanceType: {
            IntegerInstanceType: ByteStrIntMulImpl,
        },
    },
    FloorDiv: {
        IntegerInstanceType: {
            IntegerInstanceType: plt.DivideInteger,
        }
    },
    Mod: {
        IntegerInstanceType: {
            IntegerInstanceType: plt.ModInteger,
        }
    },
    Pow: {
        IntegerInstanceType: {
            IntegerInstanceType: PowImpl,
        }
    },
}

BoolOpMap = {
    And: plt.And,
    Or: plt.Or,
}

UnaryOpMap = {
    Not: {BoolInstanceType: plt.Not},
    USub: {IntegerInstanceType: lambda x: plt.SubtractInteger(plt.Integer(0), x)},
}


def rec_constant_map_data(c):
    if isinstance(c, bool):
        return uplc.PlutusInteger(int(c))
    if isinstance(c, int):
        return uplc.PlutusInteger(c)
    if isinstance(c, type(None)):
        return uplc.PlutusConstr(0, [])
    if isinstance(c, bytes):
        return uplc.PlutusByteString(c)
    if isinstance(c, str):
        return uplc.PlutusByteString(c.encode())
    if isinstance(c, list):
        return uplc.PlutusList([rec_constant_map_data(ce) for ce in c])
    if isinstance(c, dict):
        return uplc.PlutusMap(
            dict(
                zip(
                    (rec_constant_map_data(ce) for ce in c.keys()),
                    (rec_constant_map_data(ce) for ce in c.values()),
                )
            )
        )
    raise NotImplementedError(f"Unsupported constant type {type(c)}")


def rec_constant_map(c):
    if isinstance(c, bool):
        return uplc.BuiltinBool(c)
    if isinstance(c, int):
        return uplc.BuiltinInteger(c)
    if isinstance(c, type(None)):
        return uplc.BuiltinUnit()
    if isinstance(c, bytes):
        return uplc.BuiltinByteString(c)
    if isinstance(c, str):
        return uplc.BuiltinString(c)
    if isinstance(c, list):
        return uplc.BuiltinList([rec_constant_map(ce) for ce in c])
    if isinstance(c, dict):
        return uplc.BuiltinList(
            [
                uplc.BuiltinPair(*p)
                for p in zip(
                    (rec_constant_map_data(ce) for ce in c.keys()),
                    (rec_constant_map_data(ce) for ce in c.values()),
                )
            ]
        )
    if isinstance(c, PlutusData):
        return data_from_cbor(c.to_cbor())
    raise NotImplementedError(f"Unsupported constant type {type(c)}")


def wrap_validator_double_function(x: plt.AST, pass_through: int = 0):
    """
    Wraps the validator function to enable a double function as minting script

    pass_through defines how many parameters x would normally take and should be passed through to x
    """
    return plt.Lambda(
        [f"v{i}" for i in range(pass_through)] + ["a0", "a1"],
        plt.Let(
            [("p", plt.Apply(x, *(plt.Var(f"v{i}") for i in range(pass_through))))],
            plt.Ite(
                # if the second argument has constructor 0 = script context
                plt.DelayedChooseData(
                    plt.Var("a1"),
                    plt.EqualsInteger(plt.Constructor(plt.Var("a1")), plt.Integer(0)),
                    plt.Bool(False),
                    plt.Bool(False),
                    plt.Bool(False),
                    plt.Bool(False),
                ),
                # call the validator with a0, a1, and plug in "Nothing" for data
                plt.Apply(
                    plt.Var("p"),
                    plt.UPLCConstant(uplc.PlutusConstr(6, [])),
                    plt.Var("a0"),
                    plt.Var("a1"),
                ),
                # else call the validator with a0, a1 and return (now partially bound)
                plt.Apply(plt.Var("p"), plt.Var("a0"), plt.Var("a1")),
            ),
        ),
    )


def extend_statemonad(
    names: typing.List[str],
    values: typing.List[plt.AST],
    old_statemonad: plt.FunctionalMap,
):
    """Ensures that the argument is fully evaluated before being passed into the monad (like in imperative languages)"""
    assert len(names) == len(values), "Unequal amount of names and values passed in"
    lam_names = [f"a{i}" for i, _ in enumerate(names)]
    return plt.Apply(
        plt.Lambda(
            lam_names,
            plt.FunctionalMapExtend(
                old_statemonad, names, [plt.Var(n) for n in lam_names]
            ),
        ),
        *values,
    )


INITIAL_STATE = plt.FunctionalMap()


class PlutoCompiler(CompilingNodeTransformer):
    """
    Expects a TypedAST and returns UPLC/Pluto like code
    """

    step = "Compiling python statements to UPLC"

    def __init__(self, force_three_params=False, validator_function_name="validator"):
        self.force_three_params = force_three_params
        self.validator_function_name = validator_function_name

    def visit_sequence(self, node_seq: typing.List[typedstmt]) -> plt.AST:
        s = plt.Var(STATEMONAD)
        for n in node_seq:
            compiled_stmt = self.visit(n)
            s = plt.Apply(compiled_stmt, s)
        return plt.Lambda([STATEMONAD], s)

    def visit_BinOp(self, node: TypedBinOp) -> plt.AST:
        opmap = BinOpMap.get(type(node.op))
        if opmap is None:
            raise NotImplementedError(f"Operation {node.op} is not implemented")
        opmap2 = opmap.get(node.left.typ)
        if opmap2 is None:
            raise NotImplementedError(
                f"Operation {node.op} is not implemented for left type {node.left.typ}"
            )
        op = opmap2.get(node.right.typ)
        if opmap2 is None:
            raise NotImplementedError(
                f"Operation {node.op} is not implemented for left type {node.left.typ} and right type {node.right.typ}"
            )
        return plt.Lambda(
            [STATEMONAD],
            op(
                plt.Apply(self.visit(node.left), plt.Var(STATEMONAD)),
                plt.Apply(self.visit(node.right), plt.Var(STATEMONAD)),
            ),
        )

    def visit_BoolOp(self, node: TypedBoolOp) -> plt.AST:
        op = BoolOpMap.get(type(node.op))
        assert len(node.values) >= 2, "Need to compare at least to values"
        ops = op(
            plt.Apply(self.visit(node.values[0]), plt.Var(STATEMONAD)),
            plt.Apply(self.visit(node.values[1]), plt.Var(STATEMONAD)),
        )
        for v in node.values[2:]:
            ops = op(ops, plt.Apply(self.visit(v), plt.Var(STATEMONAD)))
        return plt.Lambda(
            [STATEMONAD],
            ops,
        )

    def visit_UnaryOp(self, node: TypedUnaryOp) -> plt.AST:
        opmap = UnaryOpMap.get(type(node.op))
        assert opmap is not None, f"Operator {type(node.op)} is not supported"
        op = opmap.get(node.operand.typ)
        assert (
            op is not None
        ), f"Operator {type(node.op)} is not supported for type {node.operand.typ}"
        return plt.Lambda(
            [STATEMONAD],
            op(plt.Apply(self.visit(node.operand), plt.Var(STATEMONAD))),
        )

    def visit_Compare(self, node: TypedCompare) -> plt.AST:
        assert len(node.ops) == 1, "Only single comparisons are supported"
        assert len(node.comparators) == 1, "Only single comparisons are supported"
        cmpop = node.ops[0]
        comparator = node.comparators[0].typ
        op = node.left.typ.cmp(cmpop, comparator)
        return plt.Lambda(
            [STATEMONAD],
            plt.Apply(
                op,
                plt.Apply(self.visit(node.left), plt.Var(STATEMONAD)),
                plt.Apply(self.visit(node.comparators[0]), plt.Var(STATEMONAD)),
            ),
        )

    def visit_Module(self, node: TypedModule) -> plt.AST:
        compiled_body = plt.Apply(self.visit_sequence(node.body), INITIAL_STATE)
        if self.validator_function_name is None:
            # for libraries, just return the body (a statemonad)
            validator = compiled_body
        else:
            # for validators find main function
            # TODO can use more sophisiticated procedure here i.e. functions marked by comment
            main_fun: typing.Optional[InstanceType] = None
            for s in node.body:
                if (
                    isinstance(s, FunctionDef)
                    and s.orig_name == self.validator_function_name
                ):
                    main_fun = s
            assert (
                main_fun is not None
            ), f"Could not find function named {self.validator_function_name}"
            main_fun_typ: FunctionType = main_fun.typ.typ
            assert isinstance(
                main_fun_typ, FunctionType
            ), f"Variable named {self.validator_function_name} is not of type function"

            # check if this is a contract written to double function
            enable_double_func_mint_spend = False
            if len(main_fun_typ.argtyps) >= 3 and self.force_three_params:
                # check if is possible
                second_last_arg = main_fun_typ.argtyps[-2]
                assert isinstance(
                    second_last_arg, InstanceType
                ), "Can not pass Class into validator"
                if isinstance(second_last_arg.typ, UnionType):
                    possible_types = second_last_arg.typ.typs
                else:
                    possible_types = [second_last_arg.typ]
                if any(isinstance(t, UnitType) for t in possible_types):
                    _LOGGER.warning(
                        "The redeemer is annotated to be 'None'. This value is usually encoded in PlutusData with constructor id 0 and no fields. If you want the script to double function as minting and spending script, annotate the second argument with 'NoRedeemer'."
                    )
                enable_double_func_mint_spend = not any(
                    (isinstance(t, RecordType) and t.record.constructor == 0)
                    or isinstance(t, UnitType)
                    for t in possible_types
                )
                if not enable_double_func_mint_spend:
                    _LOGGER.warning(
                        "The second argument to the validator function potentially has constructor id 0. The validator will not be able to double function as minting script and spending script."
                    )

            validator = plt.Lambda(
                [f"p{i}" for i, _ in enumerate(main_fun_typ.argtyps)] or ["_"],
                transform_output_map(main_fun_typ.rettyp)(
                    plt.Let(
                        [
                            (
                                "s",
                                compiled_body,
                            ),
                            (
                                "g",
                                plt.FunctionalMapAccess(
                                    plt.Var("s"),
                                    plt.ByteString(main_fun.name),
                                    plt.TraceError(
                                        f"NameError: {self.validator_function_name}"
                                    ),
                                ),
                            ),
                        ],
                        plt.Apply(
                            plt.Var("g"),
                            *[
                                transform_ext_params_map(a)(plt.Var(f"p{i}"))
                                for i, a in enumerate(main_fun_typ.argtyps)
                            ],
                            plt.Var("s"),
                        ),
                    ),
                ),
            )
            if enable_double_func_mint_spend:
                validator = wrap_validator_double_function(
                    validator, pass_through=len(main_fun_typ.argtyps) - 3
                )
            elif self.force_three_params:
                # Error if the double function is enforced but not possible
                raise RuntimeError(
                    "The contract can not always detect if it was passed three or two parameters on-chain."
                )
        cp = plt.Program((1, 0, 0), validator)
        return cp

    def visit_Constant(self, node: TypedConstant) -> plt.AST:
        if isinstance(node.value, bytes) and node.value != b"":
            try:
                bytes.fromhex(node.value.decode())
            except ValueError:
                pass
            else:
                _LOGGER.warning(
                    f"The string {node.value} looks like it is supposed to be a hex-encoded bytestring but is actually utf8-encoded. Try using `bytes.fromhex('{node.value.decode()}')` instead."
                )
        plt_val = plt.UPLCConstant(rec_constant_map(node.value))
        return plt.Lambda([STATEMONAD], plt_val)

    def visit_NoneType(self, _: typing.Optional[typing.Any]) -> plt.AST:
        return plt.Lambda([STATEMONAD], plt.Unit())

    def visit_Assign(self, node: TypedAssign) -> plt.AST:
        assert (
            len(node.targets) == 1
        ), "Assignments to more than one variable not supported yet"
        assert isinstance(
            node.targets[0], Name
        ), "Assignments to other things then names are not supported"
        compiled_e = self.visit(node.value)
        # (\{STATEMONAD} -> (\x -> if (x ==b {self.visit(node.targets[0])}) then ({compiled_e} {STATEMONAD}) else ({STATEMONAD} x)))
        varname = node.targets[0].id
        return plt.Lambda(
            [STATEMONAD],
            extend_statemonad(
                [varname],
                [plt.Apply(compiled_e, plt.Var(STATEMONAD))],
                plt.Var(STATEMONAD),
            ),
        )

    def visit_AnnAssign(self, node: AnnAssign) -> plt.AST:
        assert isinstance(
            node.target, Name
        ), "Assignments to other things then names are not supported"
        assert isinstance(
            node.target.typ, InstanceType
        ), "Can only assign instances to instances"
        compiled_e = self.visit(node.value)
        # (\{STATEMONAD} -> (\x -> if (x ==b {self.visit(node.targets[0])}) then ({compiled_e} {STATEMONAD}) else ({STATEMONAD} x)))
        val = plt.Apply(compiled_e, plt.Var(STATEMONAD))
        if isinstance(node.value.typ, InstanceType) and isinstance(
            node.value.typ.typ, AnyType
        ):
            # we need to map this as it will originate from PlutusData
            # AnyType is the only type other than the builtin itself that can be cast to builtin values
            val = transform_ext_params_map(node.target.typ)(val)
        if isinstance(node.target.typ, InstanceType) and isinstance(
            node.target.typ.typ, AnyType
        ):
            # we need to map this back as it will be treated as PlutusData
            # AnyType is the only type other than the builtin itself that can be cast to from builtin values
            val = transform_output_map(node.value.typ)(val)
        return plt.Lambda(
            [STATEMONAD],
            extend_statemonad(
                [node.target.id],
                [val],
                plt.Var(STATEMONAD),
            ),
        )

    def visit_Name(self, node: TypedName) -> plt.AST:
        # depending on load or store context, return the value of the variable or its name
        if not isinstance(node.ctx, Load):
            raise NotImplementedError(f"Context {node.ctx} not supported")
        if isinstance(node.typ, ClassType):
            # if this is not an instance but a class, call the constructor
            return plt.Lambda(
                [STATEMONAD],
                node.typ.constr(),
            )
        return plt.Lambda(
            [STATEMONAD],
            plt.FunctionalMapAccess(
                plt.Var(STATEMONAD),
                plt.ByteString(node.id),
                plt.TraceError(f"NameError: {node.orig_id}"),
            ),
        )

    def visit_Expr(self, node: TypedExpr) -> plt.AST:
        # we exploit UPLCs eager evaluation here
        # the expression is computed even though its value is eventually discarded
        # Note this really only makes sense for Trace
        return plt.Lambda(
            [STATEMONAD],
            plt.Apply(
                plt.Lambda(["_"], plt.Var(STATEMONAD)),
                plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
            ),
        )

    def visit_Call(self, node: TypedCall) -> plt.AST:
        # compiled_args = " ".join(f"({self.visit(a)} {STATEMONAD})" for a in node.args)
        # return rf"(\{STATEMONAD} -> ({self.visit(node.func)} {compiled_args})"
        # TODO function is actually not of type polymorphic function type here anymore
        if isinstance(node.func.typ, PolymorphicFunctionInstanceType):
            # edge case for weird builtins that are polymorphic
            func_plt = node.func.typ.polymorphic_function.impl_from_args(
                node.func.typ.typ.argtyps
            )
        else:
            func_plt = plt.Apply(self.visit(node.func), plt.Var(STATEMONAD))
        args = []
        for a, t in zip(node.args, node.func.typ.typ.argtyps):
            assert isinstance(t, InstanceType)
            # pass in all arguments evaluated with the statemonad
            a_int = plt.Apply(self.visit(a), plt.Var(STATEMONAD))
            if isinstance(t.typ, AnyType):
                # if the function expects input of generic type data, wrap data before passing it inside
                a_int = transform_output_map(a.typ)(a_int)
            args.append(a_int)
        return plt.Lambda(
            [STATEMONAD],
            plt.Apply(
                func_plt,
                *args,
                # eventually pass in the state monad as well
                plt.Var(STATEMONAD),
            ),
        )

    def visit_FunctionDef(self, node: TypedFunctionDef) -> plt.AST:
        body = node.body.copy()
        if not body or not isinstance(body[-1], Return):
            tr = Return(TypedConstant(None, typ=NoneInstanceType))
            tr.typ = NoneInstanceType
            body.append(tr)
        compiled_body = self.visit_sequence(body[:-1])
        args_state = (
            extend_statemonad(
                # the function can see its argument under the argument names
                [a.arg for a in node.args.args],
                [plt.Var(f"p{i}") for i in range(len(node.args.args))],
                plt.Var(STATEMONAD),
            )
            if node.args.args
            else plt.Var(STATEMONAD)
        )
        compiled_return = plt.Apply(
            self.visit(body[-1].value),
            plt.Apply(
                compiled_body,
                args_state,
            ),
        )
        if isinstance(node.typ.typ.rettyp.typ, AnyType):
            # if the function returns generic data, wrap the function return value
            compiled_return = transform_output_map(body[-1].value.typ)(compiled_return)
        return plt.Lambda(
            [STATEMONAD],
            extend_statemonad(
                [node.name],
                [
                    plt.Lambda(
                        # expect the statemonad again -> this is the basis for internally available values
                        [f"p{i}" for i in range(len(node.args.args))] + [STATEMONAD],
                        compiled_return,
                    )
                ],
                plt.Var(STATEMONAD),
            ),
        )

    def visit_While(self, node: TypedWhile) -> plt.AST:
        compiled_c = self.visit(node.test)
        compiled_s = self.visit_sequence(node.body)
        if node.orelse:
            # If there is orelse, transform it to an appended sequence (TODO check if this is correct)
            cn = copy(node)
            cn.orelse = []
            return self.visit_sequence([cn] + node.orelse)
        # return rf"(\{STATEMONAD} -> let g = (\s f -> if ({compiled_c} s) then f ({compiled_s} s) f else s) in (g {STATEMONAD} g))"
        return plt.Lambda(
            [STATEMONAD],
            plt.Let(
                bindings=[
                    (
                        "g",
                        plt.Lambda(
                            ["s", "f"],
                            plt.Ite(
                                plt.Apply(compiled_c, plt.Var("s")),
                                plt.Apply(
                                    plt.Var("f"),
                                    plt.Apply(compiled_s, plt.Var("s")),
                                    plt.Var("f"),
                                ),
                                plt.Var("s"),
                            ),
                        ),
                    ),
                ],
                term=plt.Apply(plt.Var("g"), plt.Var(STATEMONAD), plt.Var("g")),
            ),
        )

    def visit_For(self, node: TypedFor) -> plt.AST:
        if node.orelse:
            # If there is orelse, transform it to an appended sequence (TODO check if this is correct)
            cn = copy(node)
            cn.orelse = []
            return self.visit_sequence([cn] + node.orelse)
        assert isinstance(node.iter.typ, InstanceType)
        if isinstance(node.iter.typ.typ, ListType):
            assert isinstance(
                node.target, Name
            ), "Can only assign value to singleton element"
            return plt.Lambda(
                [STATEMONAD],
                plt.FoldList(
                    plt.Apply(self.visit(node.iter), plt.Var(STATEMONAD)),
                    plt.Lambda(
                        [STATEMONAD, "e"],
                        plt.Apply(
                            self.visit_sequence(node.body),
                            extend_statemonad(
                                [node.target.id],
                                [plt.Var("e")],
                                plt.Var(STATEMONAD),
                            ),
                        ),
                    ),
                    plt.Var(STATEMONAD),
                ),
            )
        raise NotImplementedError(
            "Compilation of for statements for anything but lists not implemented yet"
        )

    def visit_If(self, node: TypedIf) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            plt.Ite(
                plt.Apply(self.visit(node.test), plt.Var(STATEMONAD)),
                plt.Apply(self.visit_sequence(node.body), plt.Var(STATEMONAD)),
                plt.Apply(self.visit_sequence(node.orelse), plt.Var(STATEMONAD)),
            ),
        )

    def visit_Return(self, node: TypedReturn) -> plt.AST:
        raise NotImplementedError(
            "Compilation of return statements except for last statement in function is not supported."
        )

    def visit_Pass(self, node: TypedPass) -> plt.AST:
        return self.visit_sequence([])

    def visit_Subscript(self, node: TypedSubscript) -> plt.AST:
        assert isinstance(
            node.value.typ, InstanceType
        ), "Can only access elements of instances, not classes"
        if isinstance(node.value.typ.typ, TupleType):
            assert isinstance(
                node.slice, Constant
            ), "Only constant index access for tuples is supported"
            assert isinstance(
                node.slice.value, int
            ), "Only constant index integer access for tuples is supported"
            index = node.slice.value
            if index < 0:
                index += len(node.value.typ.typ.typs)
            assert isinstance(node.ctx, Load), "Tuples are read-only"
            return plt.Lambda(
                [STATEMONAD],
                plt.FunctionalTupleAccess(
                    plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
                    index,
                    len(node.value.typ.typ.typs),
                ),
            )
        if isinstance(node.value.typ.typ, PairType):
            assert isinstance(
                node.slice, Constant
            ), "Only constant index access for pairs is supported"
            assert isinstance(
                node.slice.value, int
            ), "Only constant index integer access for pairs is supported"
            index = node.slice.value
            if index < 0:
                index += 2
            assert isinstance(node.ctx, Load), "Pairs are read-only"
            assert (
                0 <= index < 2
            ), f"Pairs only have 2 elements, index should be 0 or 1, is {node.slice.value}"
            member_func = plt.FstPair if index == 0 else plt.SndPair
            # the content of pairs is always Data, so we need to unwrap
            member_typ = node.typ
            return plt.Lambda(
                [STATEMONAD],
                transform_ext_params_map(member_typ)(
                    member_func(
                        plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
                    ),
                ),
            )
        if isinstance(node.value.typ.typ, ListType):
            assert (
                node.slice.typ == IntegerInstanceType
            ), "Only single element list index access supported"
            return plt.Lambda(
                [STATEMONAD],
                plt.Let(
                    [
                        ("l", plt.Apply(self.visit(node.value), plt.Var(STATEMONAD))),
                        (
                            "raw_i",
                            plt.Apply(self.visit(node.slice), plt.Var(STATEMONAD)),
                        ),
                        (
                            "i",
                            plt.Ite(
                                plt.LessThanInteger(plt.Var("raw_i"), plt.Integer(0)),
                                plt.AddInteger(
                                    plt.Var("raw_i"), plt.LengthList(plt.Var("l"))
                                ),
                                plt.Var("raw_i"),
                            ),
                        ),
                    ],
                    plt.IndexAccessList(plt.Var("l"), plt.Var("i")),
                ),
            )
        elif isinstance(node.value.typ.typ, DictType):
            dict_typ = node.value.typ.typ
            if not isinstance(node.slice, Slice):
                return plt.Lambda(
                    [STATEMONAD],
                    plt.Let(
                        [
                            (
                                "key",
                                plt.Apply(self.visit(node.slice), plt.Var(STATEMONAD)),
                            )
                        ],
                        transform_ext_params_map(dict_typ.value_typ)(
                            plt.SndPair(
                                plt.FindList(
                                    plt.Apply(
                                        self.visit(node.value), plt.Var(STATEMONAD)
                                    ),
                                    plt.Lambda(
                                        ["x"],
                                        plt.EqualsData(
                                            transform_output_map(dict_typ.key_typ)(
                                                plt.Var("key")
                                            ),
                                            plt.FstPair(plt.Var("x")),
                                        ),
                                    ),
                                    plt.TraceError("KeyError"),
                                ),
                            ),
                        ),
                    ),
                )
        elif isinstance(node.value.typ.typ, ByteStringType):
            if not isinstance(node.slice, Slice):
                return plt.Lambda(
                    [STATEMONAD],
                    plt.Let(
                        [
                            (
                                "bs",
                                plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
                            ),
                            (
                                "raw_ix",
                                plt.Apply(self.visit(node.slice), plt.Var(STATEMONAD)),
                            ),
                            (
                                "ix",
                                plt.Ite(
                                    plt.LessThanInteger(
                                        plt.Var("raw_ix"), plt.Integer(0)
                                    ),
                                    plt.AddInteger(
                                        plt.Var("raw_ix"),
                                        plt.LengthOfByteString(plt.Var("bs")),
                                    ),
                                    plt.Var("raw_ix"),
                                ),
                            ),
                        ],
                        plt.IndexByteString(plt.Var("bs"), plt.Var("ix")),
                    ),
                )
            elif isinstance(node.slice, Slice):
                return plt.Lambda(
                    [STATEMONAD],
                    plt.Let(
                        [
                            (
                                "bs",
                                plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
                            ),
                            (
                                "raw_i",
                                plt.Apply(
                                    self.visit(node.slice.lower), plt.Var(STATEMONAD)
                                ),
                            ),
                            (
                                "i",
                                plt.Ite(
                                    plt.LessThanInteger(
                                        plt.Var("raw_i"), plt.Integer(0)
                                    ),
                                    plt.AddInteger(
                                        plt.Var("raw_i"),
                                        plt.LengthOfByteString(plt.Var("bs")),
                                    ),
                                    plt.Var("raw_i"),
                                ),
                            ),
                            (
                                "raw_j",
                                plt.Apply(
                                    self.visit(node.slice.upper), plt.Var(STATEMONAD)
                                ),
                            ),
                            (
                                "j",
                                plt.Ite(
                                    plt.LessThanInteger(
                                        plt.Var("raw_j"), plt.Integer(0)
                                    ),
                                    plt.AddInteger(
                                        plt.Var("raw_j"),
                                        plt.LengthOfByteString(plt.Var("bs")),
                                    ),
                                    plt.Var("raw_j"),
                                ),
                            ),
                            (
                                "drop",
                                plt.Ite(
                                    plt.LessThanEqualsInteger(
                                        plt.Var("i"), plt.Integer(0)
                                    ),
                                    plt.Integer(0),
                                    plt.Var("i"),
                                ),
                            ),
                            (
                                "take",
                                plt.SubtractInteger(plt.Var("j"), plt.Var("drop")),
                            ),
                        ],
                        plt.Ite(
                            plt.LessThanEqualsInteger(plt.Var("j"), plt.Var("i")),
                            plt.ByteString(b""),
                            plt.SliceByteString(
                                plt.Var("drop"),
                                plt.Var("take"),
                                plt.Var("bs"),
                            ),
                        ),
                    ),
                )
        raise NotImplementedError(
            f'Could not implement subscript "{node.slice}" of "{node.value}"'
        )

    def visit_Tuple(self, node: TypedTuple) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            plt.FunctionalTuple(
                *(plt.Apply(self.visit(e), plt.Var(STATEMONAD)) for e in node.elts)
            ),
        )

    def visit_ClassDef(self, node: TypedClassDef) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            extend_statemonad(
                [node.name],
                [node.class_typ.constr()],
                plt.Var(STATEMONAD),
            ),
        )

    def visit_Attribute(self, node: TypedAttribute) -> plt.AST:
        assert isinstance(
            node.typ, InstanceType
        ), "Can only access attributes of instances"
        obj = self.visit(node.value)
        attr = node.value.typ.attribute(node.attr)
        return plt.Lambda(
            [STATEMONAD], plt.Apply(attr, plt.Apply(obj, plt.Var(STATEMONAD)))
        )

    def visit_Assert(self, node: TypedAssert) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            plt.Ite(
                plt.Apply(self.visit(node.test), plt.Var(STATEMONAD)),
                plt.Var(STATEMONAD),
                plt.Apply(
                    plt.Error(),
                    plt.Trace(
                        plt.Apply(self.visit(node.msg), plt.Var(STATEMONAD)), plt.Unit()
                    )
                    if node.msg is not None
                    else plt.Unit(),
                ),
            ),
        )

    def visit_RawPlutoExpr(self, node: RawPlutoExpr) -> plt.AST:
        return node.expr

    def visit_List(self, node: TypedList) -> plt.AST:
        assert isinstance(node.typ, InstanceType)
        assert isinstance(node.typ.typ, ListType)
        l = empty_list(node.typ.typ.typ)
        for e in reversed(node.elts):
            l = plt.MkCons(plt.Apply(self.visit(e), plt.Var(STATEMONAD)), l)
        return plt.Lambda([STATEMONAD], l)

    def visit_Dict(self, node: TypedDict) -> plt.AST:
        assert isinstance(node.typ, InstanceType)
        assert isinstance(node.typ.typ, DictType)
        key_type = node.typ.typ.key_typ
        value_type = node.typ.typ.value_typ
        l = plt.EmptyDataPairList()
        for k, v in zip(node.keys, node.values):
            l = plt.MkCons(
                plt.MkPairData(
                    transform_output_map(key_type)(
                        plt.Apply(self.visit(k), plt.Var(STATEMONAD))
                    ),
                    transform_output_map(value_type)(
                        plt.Apply(self.visit(v), plt.Var(STATEMONAD))
                    ),
                ),
                l,
            )
        return plt.Lambda([STATEMONAD], l)

    def visit_IfExp(self, node: TypedIfExp) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            plt.Ite(
                plt.Apply(self.visit(node.test), plt.Var(STATEMONAD)),
                plt.Apply(self.visit(node.body), plt.Var(STATEMONAD)),
                plt.Apply(self.visit(node.orelse), plt.Var(STATEMONAD)),
            ),
        )

    def visit_ListComp(self, node: TypedListComp) -> plt.AST:
        assert len(node.generators) == 1, "Currently only one generator supported"
        gen = node.generators[0]
        assert isinstance(gen.iter.typ, InstanceType), "Only lists are valid generators"
        assert isinstance(gen.iter.typ.typ, ListType), "Only lists are valid generators"
        assert isinstance(
            gen.target, Name
        ), "Can only assign value to singleton element"
        lst = plt.Apply(self.visit(gen.iter), plt.Var(STATEMONAD))
        ifs = None
        for ifexpr in gen.ifs:
            if ifs is None:
                ifs = self.visit(ifexpr)
            else:
                ifs = plt.And(ifs, self.visit(ifexpr))
        map_fun = plt.Lambda(
            ["x"],
            plt.Apply(
                self.visit(node.elt),
                extend_statemonad([gen.target.id], [plt.Var("x")], plt.Var(STATEMONAD)),
            ),
        )
        empty_list_con = empty_list(node.elt.typ)
        if ifs is not None:
            filter_fun = plt.Lambda(
                ["x"],
                plt.Apply(
                    ifs,
                    extend_statemonad(
                        [gen.target.id], [plt.Var("x")], plt.Var(STATEMONAD)
                    ),
                ),
            )
            return plt.Lambda(
                [STATEMONAD],
                plt.MapFilterList(
                    lst,
                    filter_fun,
                    map_fun,
                    empty_list_con,
                ),
            )
        else:
            return plt.Lambda(
                [STATEMONAD],
                plt.MapList(
                    lst,
                    map_fun,
                    empty_list_con,
                ),
            )

    def visit_FormattedValue(self, node: TypedFormattedValue) -> plt.AST:
        return plt.Lambda(
            [STATEMONAD],
            plt.Apply(
                node.value.typ.stringify(),
                plt.Apply(self.visit(node.value), plt.Var(STATEMONAD)),
                plt.Var(STATEMONAD),
            ),
        )

    def visit_JoinedStr(self, node: TypedJoinedStr) -> plt.AST:
        joined_str = plt.Text("")
        for v in reversed(node.values):
            joined_str = plt.AppendString(
                plt.Apply(self.visit(v), plt.Var(STATEMONAD)), joined_str
            )
        return plt.Lambda([STATEMONAD], joined_str)

    def generic_visit(self, node: AST) -> plt.AST:
        raise NotImplementedError(f"Can not compile {node}")


def compile(
    prog: AST,
    filename=None,
    force_three_params=False,
    remove_dead_code=True,
    constant_folding=False,
    validator_function_name="validator",
    allow_isinstance_anything=False,
) -> plt.Program:
    compile_pipeline = [
        # Important to call this one first - it imports all further files
        RewriteImport(filename=filename),
        # Rewrites that simplify the python code
        OptimizeConstantFolding() if constant_folding else NoOp(),
        RewriteSubscript38(),
        RewriteAugAssign(),
        RewriteComparisonChaining(),
        RewriteTupleAssign(),
        RewriteImportIntegrityCheck(),
        RewriteImportPlutusData(),
        RewriteImportHashlib(),
        RewriteImportTyping(),
        RewriteForbiddenOverwrites(),
        RewriteImportDataclasses(),
        RewriteInjectBuiltins(),
        RewriteConditions(),
        # The type inference needs to be run after complex python operations were rewritten
        AggressiveTypeInferencer(allow_isinstance_anything),
        # Rewrites that circumvent the type inference or use its results
        RewriteImportUPLCBuiltins(),
        RewriteInjectBuiltinsConstr(),
        RewriteRemoveTypeStuff(),
        # Save the original names of variables
        RewriteOrigName(),
        RewriteScoping(),
        # Apply optimizations
        OptimizeRemoveDeadvars() if remove_dead_code else NoOp(),
        OptimizeVarlen(),
        OptimizeRemoveDeadconstants(),
        OptimizeRemovePass(),
        OptimizeInlineConstvars(),
    ]
    for s in compile_pipeline:
        prog = s.visit(prog)
        prog = custom_fix_missing_locations(prog)

    # the compiler runs last
    s = PlutoCompiler(
        force_three_params=force_three_params,
        validator_function_name=validator_function_name,
    )
    prog = s.visit(prog)

    return prog
