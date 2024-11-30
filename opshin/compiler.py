import copy

from uplc.ast import data_from_cbor

from .compiler_config import DEFAULT_CONFIG
from .optimize.optimize_const_folding import OptimizeConstantFolding
from .optimize.optimize_remove_comments import OptimizeRemoveDeadconstants
from .rewrite.rewrite_augassign import RewriteAugAssign
from .rewrite.rewrite_cast_condition import RewriteConditions
from .rewrite.rewrite_comparison_chaining import RewriteComparisonChaining
from .rewrite.rewrite_empty_dicts import RewriteEmptyDicts
from .rewrite.rewrite_empty_lists import RewriteEmptyLists
from .rewrite.rewrite_forbidden_overwrites import RewriteForbiddenOverwrites
from .rewrite.rewrite_forbidden_return import RewriteForbiddenReturn
from .rewrite.rewrite_import import RewriteImport
from .rewrite.rewrite_import_dataclasses import RewriteImportDataclasses
from .rewrite.rewrite_import_hashlib import RewriteImportHashlib
from .rewrite.rewrite_import_integrity_check import RewriteImportIntegrityCheck
from .rewrite.rewrite_import_plutusdata import RewriteImportPlutusData
from .rewrite.rewrite_import_typing import RewriteImportTyping
from .rewrite.rewrite_import_uplc_builtins import RewriteImportUPLCBuiltins
from .rewrite.rewrite_inject_builtins import RewriteInjectBuiltins
from .rewrite.rewrite_inject_builtin_constr import RewriteInjectBuiltinsConstr
from .rewrite.rewrite_notin import RewriteNotIn
from .rewrite.rewrite_orig_name import RewriteOrigName
from .rewrite.rewrite_remove_type_stuff import RewriteRemoveTypeStuff
from .rewrite.rewrite_scoping import RewriteScoping
from .rewrite.rewrite_subscript38 import RewriteSubscript38
from .rewrite.rewrite_tuple_assign import RewriteTupleAssign
from .optimize.optimize_remove_pass import OptimizeRemovePass
from .optimize.optimize_remove_deadvars import OptimizeRemoveDeadvars, NameLoadCollector
from .type_inference import *
from .util import (
    CompilingNodeTransformer,
    NoOp,
)
from .typed_ast import (
    transform_ext_params_map,
    transform_output_map,
    RawPlutoExpr,
)


BoolOpMap = {
    And: plt.And,
    Or: plt.Or,
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
    return OLambda(
        [f"v{i}" for i in range(pass_through)] + ["a0", "a1"],
        OLet(
            [("p", plt.Apply(x, *(OVar(f"v{i}") for i in range(pass_through))))],
            plt.Ite(
                # if the second argument has constructor 0 = script context
                plt.DelayedChooseData(
                    OVar("a1"),
                    plt.EqualsInteger(plt.Constructor(OVar("a1")), plt.Integer(0)),
                    plt.Bool(False),
                    plt.Bool(False),
                    plt.Bool(False),
                    plt.Bool(False),
                ),
                # call the validator with a0, a1, and plug in "Nothing" for data
                plt.Apply(
                    OVar("p"),
                    plt.UPLCConstant(uplc.PlutusConstr(6, [])),
                    OVar("a0"),
                    OVar("a1"),
                ),
                # else call the validator with a0, a1 and return (now partially bound)
                plt.Apply(OVar("p"), OVar("a0"), OVar("a1")),
            ),
        ),
    )


CallAST = typing.Callable[[plt.AST], plt.AST]


class PlutoCompiler(CompilingNodeTransformer):
    """
    Expects a TypedAST and returns UPLC/Pluto like code
    """

    step = "Compiling python statements to UPLC"

    def __init__(
        self,
        force_three_params=False,
        validator_function_name="validator",
        config=DEFAULT_CONFIG,
    ):
        # parameters
        self.force_three_params = force_three_params
        self.validator_function_name = validator_function_name
        self.config = config
        assert (
            self.config.fast_access_skip is None or self.config.fast_access_skip > 1
        ), "Parameter fast-access-skip needs to be greater than 1 or omitted"
        # marked knowledge during compilation
        self.current_function_typ: typing.List[FunctionType] = []

    def visit_sequence(self, node_seq: typing.List[typedstmt]) -> CallAST:
        def g(s: plt.AST):
            for n in reversed(node_seq):
                compiled_stmt = self.visit(n)
                s = compiled_stmt(s)
            return s

        return g

    def visit_BinOp(self, node: TypedBinOp) -> plt.AST:
        op = node.left.typ.binop(node.op, node.right)
        return plt.Apply(
            op,
            self.visit(node.left),
            self.visit(node.right),
        )

    def visit_BoolOp(self, node: TypedBoolOp) -> plt.AST:
        op = BoolOpMap.get(type(node.op))
        assert len(node.values) >= 2, "Need to compare at least to values"
        ops = op(
            self.visit(node.values[0]),
            self.visit(node.values[1]),
        )
        for v in node.values[2:]:
            ops = op(ops, self.visit(v))
        return ops

    def visit_UnaryOp(self, node: TypedUnaryOp) -> plt.AST:
        op = node.operand.typ.unop(node.op)
        return plt.Apply(
            op,
            self.visit(node.operand),
        )

    def visit_Compare(self, node: TypedCompare) -> plt.AST:
        assert len(node.ops) == 1, "Only single comparisons are supported"
        assert len(node.comparators) == 1, "Only single comparisons are supported"
        cmpop = node.ops[0]
        comparator = node.comparators[0].typ
        op = node.left.typ.cmp(cmpop, comparator)
        return plt.Apply(
            op,
            self.visit(node.left),
            self.visit(node.comparators[0]),
        )

    def visit_Module(self, node: TypedModule) -> plt.AST:
        # extract actually read variables by each function
        if self.validator_function_name is not None:
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
                    OPSHIN_LOGGER.warning(
                        "The redeemer is annotated to be 'None'. This value is usually encoded in PlutusData with constructor id 0 and no fields. If you want the script to double function as minting and spending script, annotate the second argument with 'NoRedeemer'."
                    )
                enable_double_func_mint_spend = not any(
                    (isinstance(t, RecordType) and t.record.constructor == 0)
                    or isinstance(t, UnitType)
                    for t in possible_types
                )
                if not enable_double_func_mint_spend:
                    OPSHIN_LOGGER.warning(
                        "The second argument to the validator function potentially has constructor id 0. The validator will not be able to double function as minting script and spending script."
                    )

            body = node.body + (
                [
                    TypedReturn(
                        TypedCall(
                            func=Name(
                                id=main_fun.name,
                                typ=InstanceType(main_fun_typ),
                                ctx=Load(),
                            ),
                            typ=main_fun_typ.rettyp,
                            args=[
                                RawPlutoExpr(
                                    expr=transform_ext_params_map(a)(
                                        OVar(f"val_param{i}")
                                    ),
                                    typ=a,
                                )
                                for i, a in enumerate(main_fun_typ.argtyps)
                            ],
                        )
                    )
                ]
            )
            self.current_function_typ.append(FunctionType([], InstanceType(AnyType())))
            name_load_visitor = NameLoadCollector()
            name_load_visitor.visit(node)
            all_vs = sorted(set(all_vars(node)) | set(name_load_visitor.loaded.keys()))

            # write all variables that are ever read
            # once at the beginning so that we can always access them (only potentially causing a nameerror at runtime)
            validator = SafeOLambda(
                [f"val_param{i}" for i, _ in enumerate(main_fun_typ.argtyps)],
                plt.Let(
                    [
                        (
                            x,
                            plt.Delay(
                                plt.TraceError(f"NameError: {map_to_orig_name(x)}")
                            ),
                        )
                        for x in all_vs
                    ],
                    self.visit_sequence(body)(
                        plt.ConstrData(plt.Integer(0), plt.EmptyDataList())
                    ),
                ),
            )
            self.current_function_typ.pop()
            if enable_double_func_mint_spend:
                validator = wrap_validator_double_function(
                    validator, pass_through=len(main_fun_typ.argtyps) - 3
                )
            elif self.force_three_params:
                # Error if the double function is enforced but not possible
                raise RuntimeError(
                    "The contract can not always detect if it was passed three or two parameters on-chain."
                )
        else:
            name_load_visitor = NameLoadCollector()
            name_load_visitor.visit(node)
            all_vs = sorted(set(all_vars(node)) | set(name_load_visitor.loaded.keys()))

            body = node.body
            # write all variables that are ever read
            # once at the beginning so that we can always access them (only potentially causing a nameerror at runtime)
            validator = plt.Let(
                [
                    (
                        x,
                        plt.Delay(plt.TraceError(f"NameError: {map_to_orig_name(x)}")),
                    )
                    for x in all_vs
                ],
                self.visit_sequence(body)(
                    plt.ConstrData(plt.Integer(0), plt.EmptyDataList())
                ),
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
                OPSHIN_LOGGER.warning(
                    f"The string {node.value} looks like it is supposed to be a hex-encoded bytestring but is actually utf8-encoded. Try using `bytes.fromhex('{node.value.decode()}')` instead."
                )
        plt_val = plt.UPLCConstant(rec_constant_map(node.value))
        return plt_val

    def visit_NoneType(self, _: typing.Optional[typing.Any]) -> plt.AST:
        return plt.Unit()

    def visit_Assign(self, node: TypedAssign) -> CallAST:
        assert (
            len(node.targets) == 1
        ), "Assignments to more than one variable not supported yet"
        assert isinstance(
            node.targets[0], Name
        ), "Assignments to other things then names are not supported"
        compiled_e = self.visit(node.value)
        varname = node.targets[0].id
        # first evaluate the term, then wrap in a delay
        return lambda x: plt.Let(
            [
                (opshin_name_scheme_compatible_varname(varname), compiled_e),
                (varname, plt.Delay(OVar(varname))),
            ],
            x,
        )

    def visit_AnnAssign(self, node: AnnAssign) -> CallAST:
        assert isinstance(
            node.target, Name
        ), "Assignments to other things then names are not supported"
        assert isinstance(
            node.target.typ, InstanceType
        ), "Can only assign instances to instances"
        val = self.visit(node.value)
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
        return lambda x: plt.Let(
            [
                (opshin_name_scheme_compatible_varname(node.target.id), val),
                (node.target.id, plt.Delay(OVar(node.target.id))),
            ],
            x,
        )

    def visit_Name(self, node: TypedName) -> plt.AST:
        # depending on load or store context, return the value of the variable or its name
        if not isinstance(node.ctx, Load):
            raise NotImplementedError(f"Context {node.ctx} not supported")
        if isinstance(node.typ, ClassType):
            # if this is not an instance but a class, call the constructor
            return node.typ.constr()
        if hasattr(node, "is_wrapped") and node.is_wrapped:
            return transform_ext_params_map(node.typ)(plt.Force(plt.Var(node.id)))
        return plt.Force(plt.Var(node.id))

    def visit_Expr(self, node: TypedExpr) -> CallAST:
        # we exploit UPLCs eager evaluation here
        # the expression is computed even though its value is eventually discarded
        # Note this really only makes sense for Trace
        # we use an invalid name here to avoid conflicts
        return lambda x: plt.Apply(OLambda(["0"], x), self.visit(node.value))

    def visit_Call(self, node: TypedCall) -> plt.AST:
        # compiled_args = " ".join(f"({self.visit(a)} {STATEMONAD})" for a in node.args)
        # return rf"(\{STATEMONAD} -> ({self.visit(node.func)} {compiled_args})"
        # TODO function is actually not of type polymorphic function type here anymore
        if isinstance(node.func.typ, PolymorphicFunctionInstanceType):
            # edge case for weird builtins that are polymorphic
            func_plt = force_params(
                node.func.typ.polymorphic_function.impl_from_args(
                    node.func.typ.typ.argtyps
                )
            )
            bind_self = None
        else:
            assert isinstance(node.func.typ, InstanceType) and isinstance(
                node.func.typ.typ, FunctionType
            )
            func_plt = self.visit(node.func)
            bind_self = node.func.typ.typ.bind_self
        bound_vs = sorted(list(node.func.typ.typ.bound_vars.keys()))
        args = []
        for i, (a, t) in enumerate(zip(node.args, node.func.typ.typ.argtyps)):
            # now impl_from_args has been chosen, skip type arg
            if (
                hasattr(node.func, "orig_id")
                and node.func.orig_id == "isinstance"
                and i == 1
            ):
                continue
            assert isinstance(t, InstanceType)
            # pass in all arguments evaluated with the statemonad
            a_int = self.visit(a)
            if isinstance(t.typ, AnyType) or isinstance(t.typ, UnionType):
                # if the function expects input of generic type data, wrap data before passing it inside
                a_int = transform_output_map(a.typ)(a_int)
            args.append(a_int)
        # First assign to let to ensure that the arguments are evaluated before the call, but need to delay
        # as this is a variable assignment
        # Also bring all states of variables read inside the function into scope / update with value in current state
        # before call to simulate statemonad with current state being passed in
        return OLet(
            [(f"p{i}", a) for i, a in enumerate(args)],
            SafeApply(
                func_plt,
                *([plt.Var(bind_self)] if bind_self is not None else []),
                *[plt.Var(n) for n in bound_vs],
                *[plt.Delay(OVar(f"p{i}")) for i in range(len(args))],
            ),
        )

    def visit_FunctionDef(self, node: TypedFunctionDef) -> CallAST:
        body = node.body.copy()
        # defaults to returning None if there is no return statement
        if node.typ.typ.rettyp.typ == AnyType():
            ret_val = plt.ConstrData(plt.Integer(0), plt.EmptyDataList())
        else:
            ret_val = plt.Unit()
        read_vs = sorted(list(node.typ.typ.bound_vars.keys()))
        if node.typ.typ.bind_self is not None:
            read_vs.insert(0, node.typ.typ.bind_self)
        self.current_function_typ.append(node.typ.typ)
        compiled_body = self.visit_sequence(body)(ret_val)
        self.current_function_typ.pop()
        return lambda x: plt.Let(
            [
                (
                    node.name,
                    plt.Delay(
                        SafeLambda(
                            read_vs + [a.arg for a in node.args.args],
                            compiled_body,
                        )
                    ),
                )
            ],
            x,
        )

    def visit_While(self, node: TypedWhile) -> CallAST:
        # the while loop calls itself, updating the values at overwritten names
        # by overwriting them with arguments to its self-recall
        if node.orelse:
            # If there is orelse, transform it to an appended sequence (TODO check if this is correct)
            cn = copy(node)
            cn.orelse = []
            return self.visit_sequence([cn] + node.orelse)
        compiled_c = self.visit(node.test)
        compiled_s = self.visit_sequence(node.body)
        written_vs = written_vars(node)
        pwritten_vs = [plt.Var(x) for x in written_vs]
        s_fun = lambda x: plt.Lambda(
            [opshin_name_scheme_compatible_varname("while")] + written_vs,
            plt.Ite(
                compiled_c,
                compiled_s(
                    plt.Apply(
                        OVar("while"),
                        OVar("while"),
                        *deepcopy(pwritten_vs),
                    )
                ),
                x,
            ),
        )

        return lambda x: OLet(
            [
                ("adjusted_next", SafeLambda(written_vs, x)),
                (
                    "while",
                    s_fun(SafeApply(OVar("adjusted_next"), *deepcopy(pwritten_vs))),
                ),
            ],
            plt.Apply(OVar("while"), OVar("while"), *deepcopy(pwritten_vs)),
        )

    def visit_For(self, node: TypedFor) -> CallAST:
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
            compiled_s = self.visit_sequence(node.body)
            compiled_iter = self.visit(node.iter)
            written_vs = written_vars(node)
            pwritten_vs = [plt.Var(x) for x in written_vs]
            s_fun = lambda x: plt.Lambda(
                [
                    opshin_name_scheme_compatible_varname("for"),
                    opshin_name_scheme_compatible_varname("iter"),
                ]
                + written_vs,
                plt.IteNullList(
                    OVar("iter"),
                    x,
                    plt.Let(
                        [(node.target.id, plt.Delay(plt.HeadList(OVar("iter"))))],
                        compiled_s(
                            plt.Apply(
                                OVar("for"),
                                OVar("for"),
                                plt.TailList(OVar("iter")),
                                *deepcopy(pwritten_vs),
                            )
                        ),
                    ),
                ),
            )
            return lambda x: OLet(
                [
                    ("adjusted_next", plt.Lambda([node.target.id] + written_vs, x)),
                    (
                        "for",
                        s_fun(
                            plt.Apply(
                                OVar("adjusted_next"),
                                plt.Var(node.target.id),
                                *deepcopy(pwritten_vs),
                            )
                        ),
                    ),
                ],
                plt.Apply(
                    OVar("for"),
                    OVar("for"),
                    compiled_iter,
                    *deepcopy(pwritten_vs),
                ),
            )
        raise NotImplementedError(
            "Compilation of for statements for anything but lists not implemented yet"
        )

    def visit_If(self, node: TypedIf) -> CallAST:
        written_vs = written_vars(node)
        pwritten_vs = [plt.Var(x) for x in written_vs]
        return lambda x: OLet(
            [("adjusted_next", SafeLambda(written_vs, x))],
            plt.Ite(
                self.visit(node.test),
                self.visit_sequence(node.body)(
                    SafeApply(OVar("adjusted_next"), *deepcopy(pwritten_vs))
                ),
                self.visit_sequence(node.orelse)(
                    SafeApply(OVar("adjusted_next"), *deepcopy(pwritten_vs))
                ),
            ),
        )

    def visit_Return(self, node: TypedReturn) -> CallAST:
        value_plt = self.visit(node.value)
        assert self.current_function_typ, "Can not handle Return outside of a function"
        if isinstance(self.current_function_typ[-1].rettyp.typ, AnyType):
            value_plt = transform_output_map(node.value.typ)(value_plt)
        return lambda _: value_plt

    def visit_Pass(self, node: TypedPass) -> CallAST:
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
            return plt.FunctionalTupleAccess(
                self.visit(node.value),
                index,
                len(node.value.typ.typ.typs),
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
            return transform_ext_params_map(member_typ)(
                member_func(
                    self.visit(node.value),
                ),
            )
        if isinstance(node.value.typ.typ, ListType):
            if not isinstance(node.slice, Slice):
                assert (
                    node.slice.typ == IntegerInstanceType
                ), "Only single element list index access supported"
                if isinstance(node.slice, Constant) and node.slice.value >= 0:
                    index = node.slice.value
                    return plt.ConstantIndexAccessListFast(
                        self.visit(node.value),
                        index,
                    )
                return OLet(
                    [
                        (
                            "l",
                            self.visit(node.value),
                        ),
                        (
                            "raw_i",
                            self.visit(node.slice),
                        ),
                        (
                            "i",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_i"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_i"), plt.LengthList(OVar("l"))
                                ),
                                OVar("raw_i"),
                            ),
                        ),
                    ],
                    (
                        plt.IndexAccessListFast(self.config.fast_access_skip)(
                            OVar("l"), OVar("i")
                        )
                        if self.config.fast_access_skip is not None
                        else plt.IndexAccessList(OVar("l"), OVar("i"))
                    ),
                )
            else:
                return OLet(
                    [
                        (
                            "xs",
                            self.visit(node.value),
                        ),
                        (
                            "raw_i",
                            self.visit(node.slice.lower),
                        ),
                        (
                            "i",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_i"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_i"),
                                    plt.LengthList(OVar("xs")),
                                ),
                                OVar("raw_i"),
                            ),
                        ),
                        (
                            "raw_j",
                            self.visit(node.slice.upper),
                        ),
                        (
                            "j",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_j"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_j"),
                                    plt.LengthList(OVar("xs")),
                                ),
                                OVar("raw_j"),
                            ),
                        ),
                        (
                            "drop",
                            plt.Ite(
                                plt.LessThanEqualsInteger(OVar("i"), plt.Integer(0)),
                                plt.Integer(0),
                                OVar("i"),
                            ),
                        ),
                        (
                            "take",
                            plt.SubtractInteger(OVar("j"), OVar("drop")),
                        ),
                    ],
                    plt.Ite(
                        plt.LessThanEqualsInteger(OVar("j"), OVar("i")),
                        empty_list(node.value.typ.typ.typ),
                        plt.SliceList(
                            OVar("drop"),
                            OVar("take"),
                            OVar("xs"),
                            empty_list(node.value.typ.typ.typ),
                        ),
                    ),
                )
        elif isinstance(node.value.typ.typ, DictType):
            dict_typ = node.value.typ.typ
            if not isinstance(node.slice, Slice):
                return OLet(
                    [
                        (
                            "key",
                            self.visit(node.slice),
                        )
                    ],
                    transform_ext_params_map(dict_typ.value_typ)(
                        plt.SndPair(
                            plt.FindList(
                                self.visit(node.value),
                                OLambda(
                                    ["x"],
                                    plt.EqualsData(
                                        transform_output_map(dict_typ.key_typ)(
                                            OVar("key")
                                        ),
                                        plt.FstPair(OVar("x")),
                                    ),
                                ),
                                plt.TraceError("KeyError"),
                            )
                        ),
                    ),
                )
        elif isinstance(node.value.typ.typ, ByteStringType):
            if not isinstance(node.slice, Slice):
                return OLet(
                    [
                        (
                            "bs",
                            self.visit(node.value),
                        ),
                        (
                            "raw_ix",
                            self.visit(node.slice),
                        ),
                        (
                            "ix",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_ix"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_ix"),
                                    plt.LengthOfByteString(OVar("bs")),
                                ),
                                OVar("raw_ix"),
                            ),
                        ),
                    ],
                    plt.IndexByteString(OVar("bs"), OVar("ix")),
                )
            elif isinstance(node.slice, Slice):
                return OLet(
                    [
                        (
                            "bs",
                            self.visit(node.value),
                        ),
                        (
                            "raw_i",
                            self.visit(node.slice.lower),
                        ),
                        (
                            "i",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_i"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_i"),
                                    plt.LengthOfByteString(OVar("bs")),
                                ),
                                OVar("raw_i"),
                            ),
                        ),
                        (
                            "raw_j",
                            self.visit(node.slice.upper),
                        ),
                        (
                            "j",
                            plt.Ite(
                                plt.LessThanInteger(OVar("raw_j"), plt.Integer(0)),
                                plt.AddInteger(
                                    OVar("raw_j"),
                                    plt.LengthOfByteString(OVar("bs")),
                                ),
                                OVar("raw_j"),
                            ),
                        ),
                        (
                            "drop",
                            plt.Ite(
                                plt.LessThanEqualsInteger(OVar("i"), plt.Integer(0)),
                                plt.Integer(0),
                                OVar("i"),
                            ),
                        ),
                        (
                            "take",
                            plt.SubtractInteger(OVar("j"), OVar("drop")),
                        ),
                    ],
                    plt.Ite(
                        plt.LessThanEqualsInteger(OVar("j"), OVar("i")),
                        plt.ByteString(b""),
                        plt.SliceByteString(
                            OVar("drop"),
                            OVar("take"),
                            OVar("bs"),
                        ),
                    ),
                )
        raise NotImplementedError(
            f'Could not implement subscript "{node.slice}" of "{node.value}"'
        )

    def visit_Tuple(self, node: TypedTuple) -> plt.AST:
        return plt.FunctionalTuple(*(self.visit(e) for e in node.elts))

    def visit_ClassDef(self, node: TypedClassDef) -> CallAST:
        return lambda x: plt.Let([(node.name, plt.Delay(node.class_typ.constr()))], x)

    def visit_Attribute(self, node: TypedAttribute) -> plt.AST:
        assert isinstance(
            node.value.typ, InstanceType
        ), "Can only access attributes of instances"
        obj = self.visit(node.value)
        attr = node.value.typ.attribute(node.attr)
        return plt.Apply(attr, obj)

    def visit_Assert(self, node: TypedAssert) -> CallAST:
        return lambda x: plt.Ite(
            self.visit(node.test),
            x,
            plt.Apply(
                plt.Error(),
                (
                    plt.Trace(self.visit(node.msg), plt.Unit())
                    if node.msg is not None
                    else plt.Unit()
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
            l = plt.MkCons(self.visit(e), l)
        return l

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
                        self.visit(k),
                    ),
                    transform_output_map(value_type)(
                        self.visit(v),
                    ),
                ),
                l,
            )
        return l

    def visit_IfExp(self, node: TypedIfExp) -> plt.AST:
        if isinstance(node.typ.typ, UnionType):
            body = self.visit(node.body)
            orelse = self.visit(node.orelse)
            if not isinstance(node.body.typ, UnionType):
                body = transform_output_map(node.body.typ)(body)
            if not isinstance(node.orelse.typ, UnionType):
                orelse = transform_output_map(node.orelse.typ)(orelse)
            return plt.Ite(self.visit(node.test), body, orelse)
        return plt.Ite(
            self.visit(node.test),
            self.visit(node.body),
            self.visit(node.orelse),
        )

    def visit_ListComp(self, node: TypedListComp) -> plt.AST:
        assert len(node.generators) == 1, "Currently only one generator supported"
        gen = node.generators[0]
        assert isinstance(gen.iter.typ, InstanceType), "Only lists are valid generators"
        assert isinstance(gen.iter.typ.typ, ListType), "Only lists are valid generators"
        assert isinstance(
            gen.target, Name
        ), "Can only assign value to singleton element"
        lst = self.visit(gen.iter)
        ifs = None
        for ifexpr in gen.ifs:
            if ifs is None:
                ifs = self.visit(ifexpr)
            else:
                ifs = plt.And(ifs, self.visit(ifexpr))
        map_fun = OLambda(
            ["x"],
            plt.Let(
                [(gen.target.id, plt.Delay(OVar("x")))],
                self.visit(node.elt),
            ),
        )
        empty_list_con = empty_list(node.elt.typ)
        if ifs is not None:
            filter_fun = OLambda(
                ["x"],
                plt.Let(
                    [(gen.target.id, plt.Delay(OVar("x")))],
                    ifs,
                ),
            )
            return plt.MapFilterList(
                lst,
                filter_fun,
                map_fun,
                empty_list_con,
            )
        else:
            return plt.MapList(
                lst,
                map_fun,
                empty_list_con,
            )

    def visit_DictComp(self, node: TypedDictComp) -> plt.AST:
        assert len(node.generators) == 1, "Currently only one generator supported"
        gen = node.generators[0]
        assert isinstance(gen.iter.typ, InstanceType), "Only lists are valid generators"
        assert isinstance(gen.iter.typ.typ, ListType), "Only lists are valid generators"
        assert isinstance(
            gen.target, Name
        ), "Can only assign value to singleton element"
        lst = self.visit(gen.iter)
        ifs = None
        for ifexpr in gen.ifs:
            if ifs is None:
                ifs = self.visit(ifexpr)
            else:
                ifs = plt.And(ifs, self.visit(ifexpr))
        map_fun = OLambda(
            ["x"],
            plt.Let(
                [(gen.target.id, plt.Delay(OVar("x")))],
                plt.MkPairData(
                    transform_output_map(node.key.typ)(
                        self.visit(node.key),
                    ),
                    transform_output_map(node.value.typ)(
                        self.visit(node.value),
                    ),
                ),
            ),
        )
        empty_list_con = plt.EmptyDataPairList()
        if ifs is not None:
            filter_fun = OLambda(
                ["x"],
                plt.Let(
                    [(gen.target.id, plt.Delay(OVar("x")))],
                    ifs,
                ),
            )
            return plt.MapFilterList(
                lst,
                filter_fun,
                map_fun,
                empty_list_con,
            )
        else:
            return plt.MapList(
                lst,
                map_fun,
                empty_list_con,
            )

    def visit_FormattedValue(self, node: TypedFormattedValue) -> plt.AST:
        return plt.Apply(
            node.value.typ.stringify(),
            self.visit(node.value),
        )

    def visit_JoinedStr(self, node: TypedJoinedStr) -> plt.AST:
        joined_str = plt.Text("")
        for v in reversed(node.values):
            joined_str = plt.AppendString(self.visit(v), joined_str)
        return joined_str

    def generic_visit(self, node: AST) -> plt.AST:
        raise NotImplementedError(f"Can not compile {node}")


def compile(
    prog: AST,
    filename=None,
    validator_function_name="validator",
    config=DEFAULT_CONFIG,
) -> plt.Program:
    compile_pipeline = [
        # Important to call this one first - it imports all further files
        RewriteImport(filename=filename),
        # Rewrites that simplify the python code
        RewriteForbiddenReturn(),
        OptimizeConstantFolding() if config.constant_folding else NoOp(),
        RewriteSubscript38(),
        RewriteAugAssign(),
        RewriteComparisonChaining(),
        RewriteTupleAssign(),
        RewriteNotIn(),
        RewriteImportIntegrityCheck(),
        RewriteImportPlutusData(),
        RewriteImportHashlib(),
        RewriteImportTyping(),
        RewriteForbiddenOverwrites(),
        RewriteImportDataclasses(),
        RewriteInjectBuiltins(),
        RewriteConditions(),
        # Save the original names of variables
        RewriteOrigName(),
        RewriteScoping(),
        # The type inference needs to be run after complex python operations were rewritten
        AggressiveTypeInferencer(config.allow_isinstance_anything),
        # Rewrites that circumvent the type inference or use its results
        RewriteEmptyLists(),
        RewriteEmptyDicts(),
        RewriteImportUPLCBuiltins(),
        RewriteInjectBuiltinsConstr(),
        RewriteRemoveTypeStuff(),
        # Apply optimizations
        OptimizeRemoveDeadvars() if config.remove_dead_code else NoOp(),
        OptimizeRemoveDeadconstants() if config.remove_dead_code else NoOp(),
        OptimizeRemovePass(),
    ]
    for s in compile_pipeline:
        prog = s.visit(prog)
        prog = custom_fix_missing_locations(prog)

    # the compiler runs last
    s = PlutoCompiler(
        force_three_params=config.force_three_params,
        validator_function_name=validator_function_name,
        config=config,
    )
    prog = s.visit(prog)

    return prog
