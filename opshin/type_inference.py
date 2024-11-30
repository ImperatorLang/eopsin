"""
An aggressive type inference based on the work of Aycock [1].
It only allows a subset of legal python operations which
allow us to infer the type of all involved variables
statically.
Using this we can resolve overloaded functions when translating Python
into UPLC where there is no dynamic type checking.
Additionally, this conveniently implements an additional layer of
security into the Smart Contract by checking type correctness.


[1]: https://legacy.python.org/workshops/2000-01/proceedings/papers/aycock/aycock.html
"""

import re

from pycardano import PlutusData
from typing import Union
from .typed_ast import *
from .util import CompilingNodeTransformer
from .fun_impls import PythonBuiltInTypes
from .rewrite.rewrite_cast_condition import SPECIAL_BOOL

# from frozendict import frozendict


INITIAL_SCOPE = {
    # class annotations
    "bytes": ByteStringType(),
    "bytearray": ByteStringType(),
    "int": IntegerType(),
    "bool": BoolType(),
    "str": StringType(),
    "Anything": AnyType(),
}

INITIAL_SCOPE.update(
    {
        name.name: typ
        for name, typ in PythonBuiltInTypes.items()
        if isinstance(typ.typ, PolymorphicFunctionType)
    }
)

DUNDER_MAP = {
    # ast.Compare:
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
    # ast.Is # no dunder
    # ast.IsNot # no dunder
    ast.In: "__contains__",
    ast.NotIn: "__contains__",
    # ast.Binop:
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.MatMult: "__matmul__",
    # ast.UnaryOp:
    # ast.UAdd
    ast.USub: "__neg__",
    ast.Not: "__bool__",
    ast.Invert: "__invert__",
    # ast.BoolOp
    ast.And: "__and__",
    ast.Or: "__or__",
}


def record_from_plutusdata(c: PlutusData):
    return Record(
        name=c.__class__.__name__,
        orig_name=c.__class__.__name__,
        constructor=c.CONSTR_ID,
        fields=frozenlist([(k, constant_type(v)) for k, v in c.__dict__.items()]),
    )


def constant_type(c):
    if isinstance(c, bool):
        return BoolInstanceType
    if isinstance(c, int):
        return IntegerInstanceType
    if isinstance(c, type(None)):
        return UnitInstanceType
    if isinstance(c, bytes):
        return ByteStringInstanceType
    if isinstance(c, str):
        return StringInstanceType
    if isinstance(c, list):
        assert len(c) > 0, "Lists must be non-empty"
        first_typ = constant_type(c[0])
        assert all(
            constant_type(ce) == first_typ for ce in c[1:]
        ), "Constant lists must contain elements of a single type only"
        return InstanceType(ListType(first_typ))
    if isinstance(c, dict):
        assert len(c) > 0, "Dicts must be non-empty"
        first_key_typ = constant_type(next(iter(c.keys())))
        first_value_typ = constant_type(next(iter(c.values())))
        assert all(
            constant_type(ce) == first_key_typ for ce in c.keys()
        ), "Constant dicts must contain keys of a single type only"
        assert all(
            constant_type(ce) == first_value_typ for ce in c.values()
        ), "Constant dicts must contain values of a single type only"
        return InstanceType(DictType(first_key_typ, first_value_typ))
    if isinstance(c, PlutusData):
        return InstanceType(RecordType(record=record_from_plutusdata(c)))
    raise NotImplementedError(f"Type {type(c)} not supported")


TypeMap = typing.Dict[str, Type]
TypeMapPair = typing.Tuple[TypeMap, TypeMap]


def union_types(*ts: Type):
    ts = OrderedSet(ts)
    # If all types are the same, just return the type
    if len(ts) == 1:
        return ts[0]
    # If there is a type that is compatible with all other types, choose the maximum
    for t in ts:
        if all(t >= tp for tp in ts):
            return t
    assert ts, "Union must combine multiple classes"
    ts = [t if isinstance(t, UnionType) else UnionType(frozenlist([t])) for t in ts]
    for e in ts:
        for e2 in e.typs:
            assert isinstance(
                e2, (RecordType, IntegerType, ByteStringType, ListType, DictType)
            ), f"Union must combine multiple PlutusData classes but found {e2.__class__.__name__}"
    union_set = OrderedSet()
    for t in ts:
        union_set.update(t.typs)
    assert distinct(
        [
            e.record.constructor
            for e in union_set
            if not isinstance(e, (ByteStringType, IntegerType, ListType, DictType))
        ]
    ), "Union must combine PlutusData classes with unique constructors"
    return UnionType(frozenlist(union_set))


def intersection_types(*ts: Type):
    ts = OrderedSet(ts)
    if len(ts) == 1:
        return ts[0]
    ts = [t if isinstance(t, UnionType) else UnionType(frozenlist([t])) for t in ts]
    assert ts, "Must have at least one type to intersect"
    intersection_set = OrderedSet(ts[0].typs)
    for t in ts[1:]:
        intersection_set.intersection_update(t.typs)
    return UnionType(frozenlist(intersection_set))


class TypeCheckVisitor(TypedNodeVisitor):
    """
    Generates the types to which objects are cast due to a boolean expression
    It returns a tuple of dictionaries which are a name -> type mapping
    for variable names that are assured to have a specific type if this expression
    is True/False respectively
    """

    def __init__(self, allow_isinstance_anything=False):
        self.allow_isinstance_anything = allow_isinstance_anything

    def generic_visit(self, node: AST) -> TypeMapPair:
        return getattr(node, "typechecks", ({}, {}))

    def visit_Call(self, node: Call) -> TypeMapPair:
        if isinstance(node.func, Name) and node.func.orig_id == SPECIAL_BOOL:
            return self.visit(node.args[0])
        if not (isinstance(node.func, Name) and node.func.orig_id == "isinstance"):
            return ({}, {})
        # special case for Union
        if not isinstance(node.args[0], Name):
            OPSHIN_LOGGER.warning(
                "Target 0 of an isinstance cast must be a variable name for type casting to work. You can still proceed, but the inferred type of the isinstance cast will not be accurate."
            )
            return ({}, {})
        assert isinstance(node.args[1], Name) or isinstance(
            node.args[1].typ, (ListType, DictType)
        ), "Target 1 of an isinstance cast must be a class name"
        target_class: RecordType = node.args[1].typ
        inst = node.args[0]
        inst_class = inst.typ
        assert isinstance(
            inst_class, InstanceType
        ), "Can only cast instances, not classes"
        # assert isinstance(target_class, RecordType), "Can only cast to PlutusData"
        if isinstance(inst_class.typ, UnionType):
            assert (
                target_class in inst_class.typ.typs
            ), f"Trying to cast an instance of Union type to non-instance of union type"
            union_without_target_class = union_types(
                *(x for x in inst_class.typ.typs if x != target_class)
            )
        elif isinstance(inst_class.typ, AnyType) and self.allow_isinstance_anything:
            union_without_target_class = AnyType()
        else:
            assert (
                inst_class.typ == target_class
            ), "Can only cast instances of Union types of PlutusData or cast the same class. If you know what you are doing, enable the flag '--allow-isinstance-anything'"
            union_without_target_class = target_class
        varname = node.args[0].id
        return ({varname: target_class}, {varname: union_without_target_class})

    def visit_BoolOp(self, node: BoolOp) -> PairType:
        res = {}
        inv_res = {}
        checks = [self.visit(v) for v in node.values]
        checked_types = defaultdict(list)
        inv_checked_types = defaultdict(list)
        for c, inv_c in checks:
            for v, t in c.items():
                checked_types[v].append(t)
            for v, t in inv_c.items():
                inv_checked_types[v].append(t)
        if isinstance(node.op, And):
            # a conjunction is just the intersection
            for v, ts in checked_types.items():
                res[v] = intersection_types(*ts)
            # if the conjunction fails, its any of the respective reverses, but only if the type is checked in every conjunction
            for v, ts in inv_checked_types.items():
                if len(ts) < len(checks):
                    continue
                inv_res[v] = union_types(*ts)
        if isinstance(node.op, Or):
            # a disjunction is just the union, but some type must be checked in every disjunction
            for v, ts in checked_types.items():
                if len(ts) < len(checks):
                    continue
                res[v] = union_types(*ts)
            # if the disjunction fails, then it must be in the intersection of the inverses
            for v, ts in inv_checked_types.items():
                inv_res[v] = intersection_types(*ts)
        return (res, inv_res)

    def visit_UnaryOp(self, node: UnaryOp) -> PairType:
        (res, inv_res) = self.visit(node.operand)
        if isinstance(node.op, Not):
            return (inv_res, res)
        return (res, inv_res)


def merge_scope(s1: typing.Dict[str, Type], s2: typing.Dict[str, Type]):
    keys = OrderedSet(s1.keys()).union(s2.keys())
    merged = {}
    for k in keys:
        if k not in s1.keys():
            merged[k] = s2[k]
        elif k not in s2.keys():
            merged[k] = s1[k]
        else:
            try:
                assert isinstance(s1[k], InstanceType) and isinstance(
                    s2[k], InstanceType
                ), "Can only merge instance types"
                merged[k] = InstanceType(union_types(s1[k].typ, s2[k].typ))
            except AssertionError as e:
                raise AssertionError(
                    f"Can not merge scopes after branching, conflicting types for {k}: {e}"
                )
    return merged


class AggressiveTypeInferencer(CompilingNodeTransformer):
    step = "Static Type Inference"

    def __init__(self, allow_isinstance_anything=False):
        self.allow_isinstance_anything = allow_isinstance_anything
        self.FUNCTION_ARGUMENT_REGISTRY = {}
        self.wrapped = []

        # A stack of dictionaries for storing scoped knowledge of variable types
        self.scopes = [INITIAL_SCOPE]

    # Obtain the type of a variable name in the current scope
    def variable_type(self, name: str) -> Type:
        name = name
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise TypeInferenceError(
            f"Variable {map_to_orig_name(name)} not initialized at access"
        )

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def set_variable_type(self, name: str, typ: Type, force=False):
        if not force and name in self.scopes[-1] and self.scopes[-1][name] != typ:
            if self.scopes[-1][name] >= typ:
                # the specified type is broader, we pass on this
                return
            raise TypeInferenceError(
                f"Type {self.scopes[-1][name]} of variable {map_to_orig_name(name)} in local scope does not match inferred type {typ}"
            )
        self.scopes[-1][name] = typ

    def implement_typechecks(self, typchecks: TypeMap):
        prevtyps = {}
        for n, t in typchecks.items():
            prevtyps[n] = self.variable_type(n).typ
            self.set_variable_type(n, InstanceType(t), force=True)
        return prevtyps

    def dunder_override(self, node: Union[BinOp, Compare, UnaryOp]):
        # Check for potential dunder_method override
        operand = None
        operation = None
        args = []
        if isinstance(node, UnaryOp):
            operand = node.operand
            operation = node.op
        elif isinstance(node, BinOp):
            operand = node.left
            operation = node.op
            args.append(node.right)
        elif isinstance(node, Compare):
            operation = node.ops[0]
            if any([isinstance(operation, x) for x in [ast.In, ast.NotIn]]):
                operand = node.comparators[0]
                args = [node.left]
            else:
                operand = node.left
                args = node.comparators
            assert len(node.ops) == 1, "Only support one op at a time"
        if operand is not None and hasattr(operand, "id"):
            operand_type = self.variable_type(operand.id)
            if (
                operation.__class__ in DUNDER_MAP
                and isinstance(operand_type, InstanceType)
                and isinstance(operand_type.typ, RecordType)
            ):
                dunder = DUNDER_MAP[operation.__class__]
                operand_class_name = operand_type.typ.record.name
                method_name = f"{operand_class_name}_{dunder}"
                if any([method_name in scope for scope in self.scopes]):
                    call = ast.Call(
                        func=ast.Attribute(
                            value=operand,
                            attr=dunder,
                            ctx=ast.Load(),
                        ),
                        args=args,
                        keywords=[],
                    )
                    call.func.orig_id = None
                    call.func.id = method_name
                    return self.visit_Call(call)
        return None

    def type_from_annotation(self, ann: expr):
        if isinstance(ann, Constant):
            if ann.value is None:
                return UnitType()
            else:
                for scope in reversed(self.scopes):
                    for key, value in scope.items():
                        if (
                            isinstance(value, RecordType)
                            and value.record.orig_name == ann.value
                        ):
                            return value

        if isinstance(ann, Name):
            if ann.id in ATOMIC_TYPES:
                return ATOMIC_TYPES[ann.id]
            if ann.id == "Self":
                v_t = self.variable_type(ann.idSelf_new)
            else:
                v_t = self.variable_type(ann.id)
            if isinstance(v_t, ClassType):
                return v_t
            raise TypeInferenceError(
                f"Class name {ann.orig_id} not initialized before annotating variable"
            )
        if isinstance(ann, Subscript):
            assert isinstance(
                ann.value, Name
            ), "Only Union, Dict and List are allowed as Generic types"
            if ann.value.orig_id == "Union":
                for elt in ann.slice.elts:
                    if isinstance(elt, Subscript) and elt.value.id == "List":
                        assert (
                            isinstance(elt.slice, Name)
                            and elt.slice.orig_id == "Anything"
                        ), f"Only List[Anything] is supported in Unions. Received List[{elt.slice.orig_id}]."
                    if isinstance(elt, Subscript) and elt.value.id == "Dict":
                        assert all(
                            isinstance(e, Name) and e.orig_id == "Anything"
                            for e in elt.slice.elts
                        ), f"Only Dict[Anything, Anything] or Dict is supported in Unions. Received Dict[{elt.slice.elts[0].orig_id}, {elt.slice.elts[1].orig_id}]."
                ann_types = frozenlist(
                    [self.type_from_annotation(e) for e in ann.slice.elts]
                )
                # check for unique constr_ids
                constr_ids = [
                    record.record.constructor
                    for record in ann_types
                    if isinstance(record, RecordType)
                ]
                assert len(constr_ids) == len(
                    set(constr_ids)
                ), f"Duplicate constr_ids for records in Union: " + str(
                    {
                        t.record.orig_name: t.record.constructor
                        for t in ann_types
                        if isinstance(t, RecordType)
                    }
                )
                return union_types(*ann_types)
            if ann.value.orig_id == "List":
                ann_type = self.type_from_annotation(ann.slice)
                assert isinstance(
                    ann_type, ClassType
                ), "List must have a single type as parameter"
                assert not isinstance(
                    ann_type, TupleType
                ), "List can currently not hold tuples"
                return ListType(InstanceType(ann_type))
            if ann.value.orig_id == "Dict":
                assert isinstance(ann.slice, Tuple), "Dict must combine two classes"
                assert len(ann.slice.elts) == 2, "Dict must combine two classes"
                ann_types = self.type_from_annotation(
                    ann.slice.elts[0]
                ), self.type_from_annotation(ann.slice.elts[1])
                assert all(
                    isinstance(e, ClassType) for e in ann_types
                ), "Dict must combine two classes"
                assert not any(
                    isinstance(e, TupleType) for e in ann_types
                ), "Dict can currently not hold tuples"
                return DictType(*(InstanceType(a) for a in ann_types))
            if ann.value.orig_id == "Tuple":
                assert isinstance(
                    ann.slice, Tuple
                ), "Tuple must combine several classes"
                ann_types = [self.type_from_annotation(e) for e in ann.slice.elts]
                assert all(
                    isinstance(e, ClassType) for e in ann_types
                ), "Tuple must combine classes"
                return TupleType(frozenlist([InstanceType(a) for a in ann_types]))
            raise NotImplementedError(
                "Only Union, Dict and List are allowed as Generic types"
            )
        if ann is None:
            return AnyType()
        raise NotImplementedError(f"Annotation type {ann.__class__} is not supported")

    def visit_sequence(self, node_seq: typing.List[stmt]) -> plt.AST:
        additional_functions = []
        for n in node_seq:
            if not isinstance(n, ast.ClassDef):
                continue
            non_method_attributes = []
            for attribute in n.body:
                if not isinstance(attribute, ast.FunctionDef):
                    non_method_attributes.append(attribute)
                    continue
                func = copy(attribute)
                if func.name[0:2] == "__" and func.name[-2:] == "__":
                    assert any(
                        [func.name == value for key, value in DUNDER_MAP.items()]
                    ), f"The following Dunder methods are supported {list(DUNDER_MAP.values())}. Received {func.name} which is not supported"
                func.name = f"{n.name}_{attribute.name}"
                for arg in func.args.args:
                    if not arg.annotation is None:
                        if isinstance(arg.annotation, ast.Name):
                            assert (
                                arg.annotation is None or arg.annotation.id != n.name
                            ), "Invalid Python, class name is undefined at this stage."
                        elif (
                            isinstance(arg.annotation, ast.Subscript)
                            and arg.annotation.value.id == "Union"
                        ):
                            for s in arg.annotation.slice.elts:
                                assert (
                                    isinstance(s, Name) and s.id != n.name
                                ) or isinstance(
                                    s, Constant
                                ), "Invalid Python, class name is undefined at this stage."
                assert isinstance(func.returns, Constant) or (
                    isinstance(func.returns, Name) and func.returns.id != n.name
                ), "Invalid Python, class name is undefined at this stage"
                ann = ast.Name(id=n.name, ctx=ast.Load())
                custom_fix_missing_locations(ann, attribute.args.args[0])
                ann.orig_id = attribute.args.args[0].orig_arg
                func.args.args[0].annotation = ann
                additional_functions.append(func)
            n.body = non_method_attributes
        if additional_functions:
            last = node_seq.pop()
            node_seq.extend(additional_functions)
            node_seq.append(last)

        stmts = []
        prevtyps = {}
        for n in node_seq:
            stmt = self.visit(n)
            stmts.append(stmt)
            # if an assert is amng the statements apply the isinstance cast
            if isinstance(stmt, Assert):
                typchecks, _ = TypeCheckVisitor(self.allow_isinstance_anything).visit(
                    stmt.test
                )
                # for the time after this assert, the variable has the specialized type
                prevtyps.update(self.implement_typechecks(typchecks))
        self.implement_typechecks(prevtyps)
        return stmts

    def visit_ClassDef(self, node: ClassDef) -> TypedClassDef:
        class_record = RecordReader.extract(node, self)
        typ = RecordType(class_record)
        self.set_variable_type(node.name, typ)
        self.FUNCTION_ARGUMENT_REGISTRY[node.name] = [
            typedarg(arg=field, typ=field_typ, orig_arg=field)
            for field, field_typ in class_record.fields
        ]
        typed_node = copy(node)
        typed_node.class_typ = typ
        return typed_node

    def visit_Constant(self, node: Constant) -> TypedConstant:
        tc = copy(node)
        assert type(node.value) not in [
            float,
            complex,
            type(...),
        ], "Float, complex numbers and ellipsis currently not supported"
        tc.typ = constant_type(node.value)
        return tc

    def visit_NoneType(self, node: None) -> TypedConstant:
        tc = Constant(value=None)
        tc.typ = constant_type(tc.value)
        return tc

    def visit_Tuple(self, node: Tuple) -> TypedTuple:
        tt = copy(node)
        tt.elts = [self.visit(e) for e in node.elts]
        tt.typ = InstanceType(TupleType(frozenlist([e.typ for e in tt.elts])))
        return tt

    def visit_List(self, node: List) -> TypedList:
        tt = copy(node)
        tt.elts = [self.visit(e) for e in node.elts]
        l_typ = tt.elts[0].typ
        assert all(
            l_typ >= e.typ for e in tt.elts
        ), "All elements of a list must have the same type"
        tt.typ = InstanceType(ListType(l_typ))
        return tt

    def visit_Dict(self, node: Dict) -> TypedDict:
        tt = copy(node)
        tt.keys = [self.visit(k) for k in node.keys]
        tt.values = [self.visit(v) for v in node.values]
        k_typ = tt.keys[0].typ
        assert all(k_typ >= k.typ for k in tt.keys), "All keys must have the same type"
        v_typ = tt.values[0].typ
        assert all(
            v_typ >= v.typ for v in tt.values
        ), "All values must have the same type"
        tt.typ = InstanceType(DictType(k_typ, v_typ))
        return tt

    def visit_Assign(self, node: Assign) -> TypedAssign:
        typed_ass = copy(node)
        typed_ass.value: TypedExpression = self.visit(node.value)
        # Make sure to first set the type of each target name so we can load it when visiting it
        for t in node.targets:
            assert isinstance(
                t, Name
            ), "Can only assign to variable names, no type deconstruction"
            # Check compatability to previous types -> variable can be bound in a function before and needs to maintain type
            self.set_variable_type(t.id, typed_ass.value.typ)
        typed_ass.targets = [self.visit(t) for t in node.targets]
        return typed_ass

    def visit_AnnAssign(self, node: AnnAssign) -> TypedAnnAssign:
        typed_ass = copy(node)
        typed_ass.annotation = self.type_from_annotation(node.annotation)
        if isinstance(typed_ass.annotation, ListType) and (
            (isinstance(node.value, Constant) and node.value.value == [])
            or (isinstance(node.value, List) and node.value.elts == [])
        ):
            # Empty lists are only allowed in annotated assignments
            typed_ass.value: TypedExpression = copy(node.value)
            typed_ass.value.typ = InstanceType(typed_ass.annotation)
        elif isinstance(typed_ass.annotation, DictType) and (
            (isinstance(node.value, Constant) and node.value.value == {})
            or (
                isinstance(node.value, Dict)
                and node.value.keys == []
                and node.value.values == []
            )
        ):
            # Empty lists are only allowed in annotated assignments
            typed_ass.value: TypedExpression = copy(node.value)
            typed_ass.value.typ = InstanceType(typed_ass.annotation)
        else:
            typed_ass.value: TypedExpression = self.visit(node.value)
        assert isinstance(
            node.target, Name
        ), "Can only assign to variable names, no type deconstruction"
        # Check compatability to previous types -> variable can be bound in a function before and needs to maintain type
        self.set_variable_type(node.target.id, InstanceType(typed_ass.annotation))
        typed_ass.target = self.visit(node.target)
        assert (
            typed_ass.value.typ >= InstanceType(typed_ass.annotation)
            or InstanceType(typed_ass.annotation) >= typed_ass.value.typ
        ), "Can only cast between related types"
        return typed_ass

    def visit_If(self, node: If) -> TypedIf:
        typed_if = copy(node)
        typed_if.test = self.visit(node.test)
        assert (
            typed_if.test.typ == BoolInstanceType
        ), "Branching condition must have boolean type"
        typchecks, inv_typchecks = TypeCheckVisitor(
            self.allow_isinstance_anything
        ).visit(typed_if.test)
        # for the time of the branch, these types are cast
        initial_scope = copy(self.scopes[-1])
        wrapped = self.implement_typechecks(typchecks)
        self.wrapped.extend(wrapped.keys())
        typed_if.body = self.visit_sequence(node.body)
        self.wrapped = [x for x in self.wrapped if x not in wrapped.keys()]

        # save resulting types
        final_scope_body = copy(self.scopes[-1])
        # reverse typechecks and remove typing of one branch
        self.scopes[-1] = initial_scope
        # for the time of the else branch, the inverse types hold
        wrapped = self.implement_typechecks(inv_typchecks)
        self.wrapped.extend(wrapped.keys())
        typed_if.orelse = self.visit_sequence(node.orelse)
        self.wrapped = [x for x in self.wrapped if x not in wrapped.keys()]
        final_scope_else = self.scopes[-1]
        # unify the resulting branch scopes
        self.scopes[-1] = merge_scope(final_scope_body, final_scope_else)
        return typed_if

    def visit_While(self, node: While) -> TypedWhile:
        typed_while = copy(node)
        typed_while.test = self.visit(node.test)
        assert (
            typed_while.test.typ == BoolInstanceType
        ), "Branching condition must have boolean type"
        typchecks, inv_typchecks = TypeCheckVisitor(
            self.allow_isinstance_anything
        ).visit(typed_while.test)
        # for the time of the branch, these types are cast
        initial_scope = copy(self.scopes[-1])
        self.implement_typechecks(typchecks)
        typed_while.body = self.visit_sequence(node.body)
        final_scope_body = copy(self.scopes[-1])
        # revert changes
        self.scopes[-1] = initial_scope
        # for the time of the else branch, the inverse types hold
        self.implement_typechecks(inv_typchecks)
        typed_while.orelse = self.visit_sequence(node.orelse)
        final_scope_else = self.scopes[-1]
        self.scopes[-1] = merge_scope(final_scope_body, final_scope_else)
        return typed_while

    def visit_For(self, node: For) -> TypedFor:
        typed_for = copy(node)
        typed_for.iter = self.visit(node.iter)
        if isinstance(node.target, Tuple):
            raise NotImplementedError(
                "Tuple deconstruction in for loops is not supported yet"
            )
        vartyp = None
        itertyp = typed_for.iter.typ
        assert isinstance(
            itertyp, InstanceType
        ), "Can only iterate over instances, not classes"
        if isinstance(itertyp.typ, TupleType):
            assert itertyp.typ.typs, "Iterating over an empty tuple is not allowed"
            vartyp = itertyp.typ.typs[0]
            assert all(
                itertyp.typ.typs[0] == t for t in typed_for.iter.typ.typs
            ), "Iterating through a tuple requires the same type for each element"
        elif isinstance(itertyp.typ, ListType):
            vartyp = itertyp.typ.typ
        else:
            raise NotImplementedError(
                "Type inference for loops over non-list objects is not supported"
            )
        self.set_variable_type(node.target.id, vartyp)
        typed_for.target = self.visit(node.target)
        typed_for.body = self.visit_sequence(node.body)
        typed_for.orelse = self.visit_sequence(node.orelse)
        return typed_for

    def visit_Name(self, node: Name) -> TypedName:
        tn = copy(node)
        # typing List and Dict are not present in scope we don't want to call variable_type
        if node.orig_id == "List":
            tn.typ = ListType(InstanceType(AnyType()))
        elif node.orig_id == "Dict":
            tn.typ = DictType(InstanceType(AnyType()), InstanceType(AnyType()))
        else:
            # Make sure that the rhs of an assign is evaluated first
            tn.typ = self.variable_type(node.id)
        if node.id in self.wrapped:
            tn.is_wrapped = True
        return tn

    def visit_keyword(self, node: keyword) -> Typedkeyword:
        tk = copy(node)
        tk.value = self.visit(node.value)
        return tk

    def visit_Compare(self, node: Compare) -> Union[TypedCompare, TypedCall]:
        dunder_node = self.dunder_override(node)
        if dunder_node is not None:
            if isinstance(node.ops[0], ast.NotIn):
                return self.visit(ast.UnaryOp(op=ast.Not(), operand=dunder_node))
            return dunder_node
        typed_cmp = copy(node)
        typed_cmp.left = self.visit(node.left)
        typed_cmp.comparators = [self.visit(s) for s in node.comparators]
        typed_cmp.typ = BoolInstanceType

        return typed_cmp

    def visit_arg(self, node: arg) -> typedarg:
        ta = copy(node)
        ta.typ = InstanceType(self.type_from_annotation(node.annotation))
        self.set_variable_type(ta.arg, ta.typ)
        return ta

    def visit_arguments(self, node: arguments) -> typedarguments:
        if node.kw_defaults or node.kwarg or node.kwonlyargs or node.defaults:
            raise NotImplementedError(
                "Keyword arguments and defaults not supported yet"
            )
        ta = copy(node)
        ta.args = [self.visit(a) for a in node.args]
        return ta

    def visit_FunctionDef(self, node: FunctionDef) -> TypedFunctionDef:
        tfd = copy(node)
        wraps_builtin = (
            all(
                isinstance(o, Name) and o.orig_id == "wraps_builtin"
                for o in node.decorator_list
            )
            and node.decorator_list
        )
        assert (
            not node.decorator_list or wraps_builtin
        ), "Functions may not have decorators other than wraps_builtin"
        for i, arg in enumerate(node.args.args):
            if hasattr(arg.annotation, "idSelf"):
                tfd.args.args[i].annotation.id = tfd.args.args[0].annotation.id
        if hasattr(node.returns, "idSelf"):
            tfd.returns.id = tfd.args.args[0].annotation.id

        self.enter_scope()
        tfd.args = self.visit(node.args)

        functyp = FunctionType(
            frozenlist([t.typ for t in tfd.args.args]),
            InstanceType(self.type_from_annotation(tfd.returns)),
            bound_vars={
                v: self.variable_type(v)
                for v in externally_bound_vars(node)
                if not v in ["List", "Dict"]
            },
            bind_self=node.name if node.name in read_vars(node) else None,
        )
        tfd.typ = InstanceType(functyp)
        if wraps_builtin:
            # the body of wrapping builtin functions is fully ignored
            pass
        else:
            # We need the function type inside for recursion
            self.set_variable_type(node.name, tfd.typ)
            tfd.body = self.visit_sequence(node.body)
            # Check that return type and annotated return type match
            rets_extractor = ReturnExtractor(functyp.rettyp)
            rets_extractor.check_fulfills(tfd)

        self.exit_scope()
        # We need the function type outside for usage
        self.set_variable_type(node.name, tfd.typ)
        self.FUNCTION_ARGUMENT_REGISTRY[node.name] = node.args.args
        return tfd

    def visit_Module(self, node: Module) -> TypedModule:
        self.enter_scope()
        tm = copy(node)
        tm.body = self.visit_sequence(node.body)
        self.exit_scope()
        return tm

    def visit_Expr(self, node: Expr) -> TypedExpr:
        tn = copy(node)
        tn.value = self.visit(node.value)
        return tn

    def visit_BinOp(self, node: BinOp) -> Union[TypedBinOp, TypedCall]:
        dunder_node = self.dunder_override(node)
        if dunder_node is not None:
            return dunder_node
        tb = copy(node)
        tb.left = self.visit(node.left)
        tb.right = self.visit(node.right)
        binop_fun_typ: FunctionType = tb.left.typ.binop_type(tb.op, tb.right.typ)
        tb.typ = binop_fun_typ.rettyp

        return tb

    def visit_BoolOp(self, node: BoolOp) -> TypedBoolOp:
        tt = copy(node)
        if isinstance(node.op, And):
            values = []
            prevtyps = {}
            for e in node.values:
                values.append(self.visit(e))
                typchecks, _ = TypeCheckVisitor(self.allow_isinstance_anything).visit(
                    values[-1]
                )
                # for the time after the shortcut and the variable type to the specialized type
                prevtyps.update(self.implement_typechecks(typchecks))
            self.implement_typechecks(prevtyps)
            tt.values = values
        elif isinstance(node.op, Or):
            values = []
            prevtyps = {}
            for e in node.values:
                values.append(self.visit(e))
                _, inv_typechecks = TypeCheckVisitor(
                    self.allow_isinstance_anything
                ).visit(values[-1])
                # for the time after the shortcut or the variable type is *not* the specialized type
                prevtyps.update(self.implement_typechecks(inv_typechecks))
            self.implement_typechecks(prevtyps)
            tt.values = values
        else:
            tt.values = [self.visit(e) for e in node.values]
        tt.typ = BoolInstanceType
        assert all(
            BoolInstanceType >= e.typ for e in tt.values
        ), "All values compared must be bools"
        return tt

    def visit_UnaryOp(self, node: UnaryOp) -> TypedUnaryOp:
        dunder_node = self.dunder_override(node)
        if dunder_node is not None:
            if isinstance(node.op, ast.Not):
                node.operand = dunder_node
            else:
                return dunder_node
        tu = copy(node)
        tu.operand = self.visit(node.operand)
        tu.typ = tu.operand.typ.typ.unop_type(node.op).rettyp
        return tu

    def visit_Subscript(self, node: Subscript) -> TypedSubscript:
        ts = copy(node)
        # special case: Subscript of Union / Dict / List and atomic types
        if isinstance(ts.value, Name) and ts.value.orig_id in [
            "Union",
            "Dict",
            "List",
        ]:
            ts.value = ts.typ = self.type_from_annotation(ts)
            return ts

        ts.value = self.visit(node.value)
        assert isinstance(ts.value.typ, InstanceType), "Can only subscript instances"
        if isinstance(ts.value.typ.typ, TupleType):
            assert (
                ts.value.typ.typ.typs
            ), "Accessing elements from the empty tuple is not allowed"
            if all(ts.value.typ.typ.typs[0] == t for t in ts.value.typ.typ.typs):
                ts.typ = ts.value.typ.typ.typs[0]
            elif isinstance(ts.slice, Constant) and isinstance(ts.slice.value, int):
                ts.typ = ts.value.typ.typ.typs[ts.slice.value]
            else:
                raise TypeInferenceError(
                    f"Could not infer type of subscript of typ {ts.value.typ.typ.__class__}"
                )
        elif isinstance(ts.value.typ.typ, PairType):
            if isinstance(ts.slice, Constant) and isinstance(ts.slice.value, int):
                ts.typ = (
                    ts.value.typ.typ.l_typ
                    if ts.slice.value == 0
                    else ts.value.typ.typ.r_typ
                )
            else:
                raise TypeInferenceError(
                    f"Could not infer type of subscript of typ {ts.value.typ.typ.__class__}"
                )
        elif isinstance(ts.value.typ.typ, ListType):
            if not isinstance(ts.slice, Slice):
                ts.typ = ts.value.typ.typ.typ
                ts.slice = self.visit(node.slice)
                assert (
                    ts.slice.typ == IntegerInstanceType
                ), "List indices must be integers"
            else:
                ts.typ = ts.value.typ
                if ts.slice.lower is None:
                    ts.slice.lower = Constant(0)
                ts.slice.lower = self.visit(node.slice.lower)
                assert (
                    ts.slice.lower.typ == IntegerInstanceType
                ), "lower slice indices for lists must be integers"
                if ts.slice.upper is None:
                    ts.slice.upper = Call(
                        func=Name(id="len", ctx=Load()), args=[ts.value], keywords=[]
                    )
                    ts.slice.upper.func.orig_id = "len"
                ts.slice.upper = self.visit(node.slice.upper)
                assert (
                    ts.slice.upper.typ == IntegerInstanceType
                ), "upper slice indices for lists must be integers"
        elif isinstance(ts.value.typ.typ, ByteStringType):
            if not isinstance(ts.slice, Slice):
                ts.typ = IntegerInstanceType
                ts.slice = self.visit(node.slice)
                assert (
                    ts.slice.typ == IntegerInstanceType
                ), "bytes indices must be integers"
            else:
                ts.typ = ByteStringInstanceType
                if ts.slice.lower is None:
                    ts.slice.lower = Constant(0)
                ts.slice.lower = self.visit(node.slice.lower)
                assert (
                    ts.slice.lower.typ == IntegerInstanceType
                ), "lower slice indices for bytes must be integers"
                if ts.slice.upper is None:
                    ts.slice.upper = Call(
                        func=Name(id="len", ctx=Load()), args=[ts.value], keywords=[]
                    )
                    ts.slice.upper.func.orig_id = "len"
                ts.slice.upper = self.visit(node.slice.upper)
                assert (
                    ts.slice.upper.typ == IntegerInstanceType
                ), "upper slice indices for bytes must be integers"
        elif isinstance(ts.value.typ.typ, DictType):
            if not isinstance(ts.slice, Slice):
                ts.slice = self.visit(node.slice)
                assert (
                    ts.slice.typ == ts.value.typ.typ.key_typ
                ), f"Dict subscript must have dict key type {ts.value.typ.typ.key_typ} but has type {ts.slice.typ}"
                ts.typ = ts.value.typ.typ.value_typ
            else:
                raise TypeInferenceError(
                    f"Could not infer type of subscript of dict with a slice."
                )
        else:
            raise TypeInferenceError(
                f"Could not infer type of subscript of typ {ts.value.typ.__class__}"
            )
        return ts

    def visit_Call(self, node: Call) -> TypedCall:
        tc = copy(node)
        if node.keywords:
            assert (
                node.func.id in self.FUNCTION_ARGUMENT_REGISTRY
            ), "Keyword arguments can only be used with user defined functions"
            keywords = copy(node.keywords)
            reg_args = self.FUNCTION_ARGUMENT_REGISTRY[node.func.id]
            args = []
            for i, a in enumerate(reg_args):
                if len(node.args) > i:
                    args.append(self.visit(node.args[i]))
                else:
                    candidates = [
                        (idx, keyword)
                        for idx, keyword in enumerate(keywords)
                        if keyword.arg == a.orig_arg
                    ]
                    assert (
                        len(candidates) == 1
                    ), f"There should be one keyword or positional argument for the arg {a.orig_arg} but found {len(candidates)}"
                    args.append(self.visit(candidates[0][1].value))
                    keywords.pop(candidates[0][0])
            assert (
                len(keywords) == 0
            ), f"Could not match the keywords {[keyword.arg for keyword in keywords]} to any argument"
            tc.args = args
            tc.keywords = []
        else:
            tc.args = [self.visit(a) for a in node.args]

        # might be isinstance
        # Subscripts are not allowed in isinstance calls
        if (
            isinstance(tc.func, Name)
            and tc.func.orig_id == "isinstance"
            and isinstance(tc.args[1], Subscript)
        ):
            raise TypeError(
                "Subscripted generics cannot be used with class and instance checks"
            )

        # Need to handle the presence of PlutusData classes
        if (
            isinstance(tc.func, Name)
            and tc.func.orig_id == "isinstance"
            and not isinstance(
                tc.args[1].typ, (ByteStringType, IntegerType, ListType, DictType)
            )
            and not hasattr(node, "skip_next")
        ):
            target_class = tc.args[1].typ
            if (
                isinstance(tc.args[0].typ, InstanceType)
                and isinstance(tc.args[0].typ.typ, AnyType)
                and not self.allow_isinstance_anything
            ):
                raise AssertionError(
                    "OpShin does not permit checking the instance of raw Anything/Datum objects as this only checks the equality of the constructor id and nothing more. "
                    "If you are certain of what you are doing, please use the flag '--allow-isinstance-anything'."
                )
            ntc = Compare(
                left=Attribute(tc.args[0], "CONSTR_ID"),
                ops=[Eq()],
                comparators=[Constant(target_class.record.constructor)],
            )
            custom_fix_missing_locations(ntc, node)
            ntc = self.visit(ntc)
            ntc.typ = BoolInstanceType
            ntc.typechecks = TypeCheckVisitor(self.allow_isinstance_anything).visit(tc)
            if isinstance(tc.args[0].typ.typ, UnionType) and any(
                [
                    isinstance(a, (IntegerType, ByteStringType, ListType, DictType))
                    for a in tc.args[0].typ.typ.typs
                ]
            ):
                n = copy(node)
                n.skip_next = True
                return self.visit(BoolOp(And(), [n, ntc]))
            else:
                return ntc
        try:
            tc.func = self.visit(node.func)
        except Exception as e:
            # might be a method, duck test for class_name, method_name should
            try:
                func_variable_type = self.variable_type(tc.func.value.id)
                class_name = func_variable_type.typ.record.name
                method_name = f"{class_name}_{tc.func.attr}"
                # If method_name found then use this.
                self.variable_type(method_name)
            except Exception:
                # if this fails raise original error
                raise e
            n = ast.Name(id=method_name, ctx=ast.Load())
            n.orig_id = node.func.attr
            tc.func = self.visit(n)
            tc.func.orig_id = node.func.attr
            c_self = ast.Name(id=node.func.value.id, ctx=ast.Load())
            c_self.orig_id = None
            tc.args.insert(0, self.visit(c_self))

        # might be a class
        if isinstance(tc.func.typ, ClassType):
            tc.func.typ = tc.func.typ.constr_type()
        # type might only turn out after the initialization (note the constr could be polymorphic)
        if isinstance(tc.func.typ, InstanceType) and isinstance(
            tc.func.typ.typ, PolymorphicFunctionType
        ):
            tc.func.typ = PolymorphicFunctionInstanceType(
                tc.func.typ.typ.polymorphic_function.type_from_args(
                    [a.typ for a in tc.args]
                ),
                tc.func.typ.typ.polymorphic_function,
            )
        if isinstance(tc.func.typ, InstanceType) and isinstance(
            tc.func.typ.typ, FunctionType
        ):
            functyp = tc.func.typ.typ
            assert len(tc.args) == len(
                functyp.argtyps
            ), f"Signature of function does not match number of arguments. Expected {len(functyp.argtyps)} arguments with these types: {functyp.argtyps} but got {len(tc.args)} arguments."
            # all arguments need to be subtypes of the parameter type
            for i, (a, ap) in enumerate(zip(tc.args, functyp.argtyps)):
                assert (
                    ap >= a.typ
                ), f"Signature of function does not match arguments in argument {i}. Expected this type: {ap} but got {a.typ}."
            tc.typ = functyp.rettyp
            return tc
        raise TypeInferenceError("Could not infer type of call")

    def visit_Pass(self, node: Pass) -> TypedPass:
        tp = copy(node)
        return tp

    def visit_Return(self, node: Return) -> TypedReturn:
        tp = copy(node)
        tp.value = self.visit(node.value)
        tp.typ = tp.value.typ
        return tp

    def visit_Attribute(self, node: Attribute) -> TypedAttribute:
        tp = copy(node)
        tp.value = self.visit(node.value)
        owner = tp.value.typ
        # accesses to field
        tp.typ = owner.attribute_type(node.attr)
        return tp

    def visit_Assert(self, node: Assert) -> TypedAssert:
        ta = copy(node)
        ta.test = self.visit(node.test)
        try:
            warn_assert_msg = f" (see assert with message '{node.msg.values[0].value}')"
        except AttributeError:
            warn_assert_msg = ""
        if isinstance(ta.test.args[0], Constant) and ta.test.args[0].value is None:
            OPSHIN_LOGGER.warning(
                "Asserting `None'"
                + warn_assert_msg
                + " is equivalent to asserting False, which always fails."
            )
        elif isinstance(ta.test.args[0].typ.typ, UnitType):
            OPSHIN_LOGGER.warning(
                "Asserting `None'"
                + warn_assert_msg
                + " is likely to stem from a procedure already doing internal assertions and returning `None'. Asserting this is equivalent to asserting False, which always fails (likely unintended)."
            )
        assert (
            ta.test.typ == BoolInstanceType
        ), "Assertions must result in a boolean type"
        if ta.msg is not None:
            ta.msg = self.visit(node.msg)
            assert (
                ta.msg.typ == StringInstanceType
            ), "Assertions must has a string message (or None)"
        return ta

    def visit_RawPlutoExpr(self, node: RawPlutoExpr) -> RawPlutoExpr:
        assert node.typ is not None, "Raw Pluto Expression is missing type annotation"
        return node

    def visit_IfExp(self, node: IfExp) -> TypedIfExp:
        node_cp = copy(node)
        node_cp.test = self.visit(node.test)
        assert node_cp.test.typ == BoolInstanceType, "Comparison must have type boolean"
        typchecks, inv_typchecks = TypeCheckVisitor(
            self.allow_isinstance_anything
        ).visit(node_cp.test)
        prevtyps = self.implement_typechecks(typchecks)
        self.wrapped.extend(prevtyps.keys())
        node_cp.body = self.visit(node.body)
        self.wrapped = [x for x in self.wrapped if x not in prevtyps.keys()]

        self.implement_typechecks(prevtyps)
        prevtyps = self.implement_typechecks(inv_typchecks)
        self.wrapped.extend(prevtyps.keys())
        node_cp.orelse = self.visit(node.orelse)
        self.wrapped = [x for x in self.wrapped if x not in prevtyps.keys()]
        self.implement_typechecks(prevtyps)
        if node_cp.body.typ >= node_cp.orelse.typ:
            node_cp.typ = node_cp.body.typ
        elif node_cp.orelse.typ >= node_cp.body.typ:
            node_cp.typ = node_cp.orelse.typ
        else:
            try:
                assert isinstance(node_cp.body.typ, InstanceType) and isinstance(
                    node_cp.orelse.typ, InstanceType
                )
                node_cp.typ = InstanceType(
                    union_types(node_cp.body.typ.typ, node_cp.orelse.typ.typ)
                )
            except AssertionError:
                raise TypeInferenceError(
                    "Branches of if-expression must return compatible types."
                )
        return node_cp

    def visit_comprehension(self, g: comprehension) -> typedcomprehension:
        new_g = copy(g)
        if isinstance(g.target, Tuple):
            raise NotImplementedError(
                "Type deconstruction in for loops is not supported yet"
            )
        new_g.iter = self.visit(g.iter)
        itertyp = new_g.iter.typ
        assert isinstance(
            itertyp, InstanceType
        ), "Can only iterate over instances, not classes"
        if isinstance(itertyp.typ, TupleType):
            assert itertyp.typ.typs, "Iterating over an empty tuple is not allowed"
            vartyp = itertyp.typ.typs[0]
            assert all(
                itertyp.typ.typs[0] == t for t in new_g.iter.typ.typs
            ), "Iterating through a tuple requires the same type for each element"
        elif isinstance(itertyp.typ, ListType):
            vartyp = itertyp.typ.typ
        else:
            raise NotImplementedError(
                "Type inference for loops over non-list objects is not supported"
            )
        self.set_variable_type(g.target.id, vartyp)
        new_g.target = self.visit(g.target)
        new_g.ifs = [self.visit(i) for i in g.ifs]
        return new_g

    def visit_ListComp(self, node: ListComp) -> TypedListComp:
        typed_listcomp = copy(node)
        # inside the comprehension is a seperate scope
        self.enter_scope()
        # first evaluate generators for assigned variables
        typed_listcomp.generators = [self.visit(s) for s in node.generators]
        # then evaluate elements
        typed_listcomp.elt = self.visit(node.elt)
        self.exit_scope()
        typed_listcomp.typ = InstanceType(ListType(typed_listcomp.elt.typ))
        return typed_listcomp

    def visit_DictComp(self, node: DictComp) -> TypedDictComp:
        typed_dictcomp = copy(node)
        # inside the comprehension is a seperate scope
        self.enter_scope()
        # first evaluate generators for assigned variables
        typed_dictcomp.generators = [self.visit(s) for s in node.generators]
        # then evaluate elements
        typed_dictcomp.key = self.visit(node.key)
        typed_dictcomp.value = self.visit(node.value)
        self.exit_scope()
        typed_dictcomp.typ = InstanceType(
            DictType(typed_dictcomp.key.typ, typed_dictcomp.value.typ)
        )
        return typed_dictcomp

    def visit_FormattedValue(self, node: FormattedValue) -> TypedFormattedValue:
        typed_node = copy(node)
        typed_node.value = self.visit(node.value)
        assert node.conversion in (
            -1,
            115,
        ), "Only string formatting is allowed but got repr or ascii formatting."
        assert (
            node.format_spec is None
        ), "No format specification is allowed but got formatting specifiers (i.e. decimals)."
        typed_node.typ = StringInstanceType
        return typed_node

    def visit_JoinedStr(self, node: JoinedStr) -> TypedJoinedStr:
        typed_node = copy(node)
        typed_node.values = [self.visit(v) for v in node.values]
        typed_node.typ = StringInstanceType
        return typed_node

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        assert node.module == "opshin.bridge", "Trying to import from invalid location"
        return node

    def generic_visit(self, node: AST) -> TypedAST:
        raise NotImplementedError(
            f"Cannot infer type of non-implemented node {node.__class__}"
        )


class RecordReader(NodeVisitor):
    name: str
    orig_name: str
    constructor: typing.Optional[int]
    attributes: typing.List[typing.Tuple[str, Type]]
    _type_inferencer: AggressiveTypeInferencer

    def __init__(self, type_inferencer: AggressiveTypeInferencer):
        self.constructor = None
        self.attributes = []
        self._type_inferencer = type_inferencer

    @classmethod
    def extract(cls, c: ClassDef, type_inferencer: AggressiveTypeInferencer) -> Record:
        f = cls(type_inferencer)
        f.visit(c)
        if f.constructor is None:
            det_string = RecordType(
                Record(f.name, f.orig_name, 0, frozenlist(f.attributes))
            ).id_map(skip_constructor=True)
            det_hash = sha256(str(det_string).encode("utf8")).hexdigest()
            f.constructor = int(det_hash, 16) % 2**32
        return Record(f.name, f.orig_name, f.constructor, frozenlist(f.attributes))

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        assert isinstance(
            node.target, Name
        ), "Record elements must have named attributes"
        typ = self._type_inferencer.type_from_annotation(node.annotation)
        if node.target.id != "CONSTR_ID":
            assert (
                node.value is None
            ), f"PlutusData attribute {node.target.id} may not have a default value"
            assert not isinstance(
                typ, TupleType
            ), "Records can currently not hold tuples"
            self.attributes.append(
                (
                    node.target.id,
                    InstanceType(typ),
                )
            )
            return
        assert typ == IntegerType, "CONSTR_ID must be assigned an integer"
        assert isinstance(
            node.value, Constant
        ), "CONSTR_ID must be assigned a constant integer"
        assert isinstance(
            node.value.value, int
        ), "CONSTR_ID must be assigned an integer"
        self.constructor = node.value.value

    def visit_ClassDef(self, node: ClassDef) -> None:
        self.name = node.name
        self.orig_name = node.orig_name
        for s in node.body:
            self.visit(s)

    def visit_Pass(self, node: Pass) -> None:
        pass

    def visit_Assign(self, node: Assign) -> None:
        assert len(node.targets) == 1, "Record elements must be assigned one by one"
        target = node.targets[0]
        assert isinstance(target, Name), "Record elements must have named attributes"
        assert (
            target.id == "CONSTR_ID"
        ), "Type annotations may only be omitted for CONSTR_ID"
        assert isinstance(
            node.value, Constant
        ), "CONSTR_ID must be assigned a constant integer"
        assert isinstance(
            node.value.value, int
        ), "CONSTR_ID must be assigned an integer"
        self.constructor = node.value.value

    def visit_Expr(self, node: Expr) -> None:
        assert isinstance(
            node.value, Constant
        ), "Only comments are allowed inside classes"
        return None

    def generic_visit(self, node: AST) -> None:
        raise NotImplementedError(f"Can not compile {ast.dump(node)} inside of a class")


def typed_ast(ast: AST):
    return AggressiveTypeInferencer().visit(ast)


def map_to_orig_name(name: str):
    return re.sub(r"_\d+$", "", name)


class ReturnExtractor(TypedNodeVisitor):
    """
    Utility to check that all paths end in Return statements with the proper type

    Returns whether there is no remaining path
    """

    def __init__(self, func_rettyp: Type):
        self.func_rettyp = func_rettyp

    def visit_sequence(self, nodes: typing.List[TypedAST]) -> bool:
        all_paths_covered = False
        for node in nodes:
            all_paths_covered = self.visit(node)
            if all_paths_covered:
                break
        return all_paths_covered

    def visit_If(self, node: If) -> bool:
        return self.visit_sequence(node.body) and self.visit_sequence(node.orelse)

    def visit_For(self, node: For) -> bool:
        # The body simply has to be checked but has no influence on whether all paths are covered
        # because it might never be visited
        self.visit_sequence(node.body)
        # the else path is always visited
        return self.visit_sequence(node.orelse)

    def visit_While(self, node: For) -> bool:
        # The body simply has to be checked but has no influence on whether all paths are covered
        # because it might never be visited
        self.visit_sequence(node.body)
        # the else path is always visited
        return self.visit_sequence(node.orelse)

    def visit_Return(self, node: Return) -> bool:
        assert (
            self.func_rettyp >= node.typ
        ), f"Function annotated return type does not match actual return type"
        return True

    def check_fulfills(self, node: FunctionDef):
        all_paths_covered = self.visit_sequence(node.body)
        if not all_paths_covered:
            assert (
                self.func_rettyp >= NoneInstanceType
            ), f"Function '{node.name}' has no return statement but is supposed to return not-None value"
