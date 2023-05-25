import typing
from collections import defaultdict

from copy import copy

from ast import *

try:
    unparse
except NameError:
    from astunparse import unparse

from ..util import CompilingNodeTransformer, CompilingNodeVisitor
from ..type_inference import INITIAL_SCOPE

"""
Pre-evaluates constant statements
"""

ACCEPTED_ATOMIC_TYPES = [
    int,
    str,
    bytes,
    type(None),
    bool,
]

SAFE_GLOBALS_LIST = [
    abs,
    all,
    any,
    ascii,
    bin,
    bool,
    bytes,
    bytearray,
    callable,
    chr,
    classmethod,
    compile,
    complex,
    delattr,
    dict,
    dir,
    divmod,
    enumerate,
    filter,
    float,
    format,
    frozenset,
    getattr,
    hasattr,
    hash,
    hex,
    id,
    input,
    int,
    isinstance,
    issubclass,
    iter,
    len,
    list,
    map,
    max,
    min,
    next,
    object,
    oct,
    open,
    ord,
    pow,
    print,
    property,
    range,
    repr,
    reversed,
    round,
    set,
    setattr,
    slice,
    sorted,
    staticmethod,
    str,
    sum,
    super,
    tuple,
    type,
    vars,
    zip,
]
SAFE_GLOBALS = {x.__name__: x for x in SAFE_GLOBALS_LIST}


class ShallowNameDefCollector(CompilingNodeVisitor):
    step = "Collecting occuring variable names"

    def __init__(self):
        self.vars = set()

    def visit_Name(self, node: Name) -> None:
        if isinstance(node.ctx, Store):
            self.vars.add(node.id)

    def visit_ClassDef(self, node: ClassDef):
        self.vars.add(node.name)
        # ignore the content (i.e. attribute names) of class definitions

    def visit_FunctionDef(self, node: FunctionDef):
        self.vars.add(node.name)
        # ignore the recursive stuff


class DefinedTimesVisitor(CompilingNodeVisitor):
    step = "Collecting how often variables are written"

    def __init__(self):
        self.vars = defaultdict(int)

    def visit_For(self, node: For) -> None:
        return
        # TODO future items: use this together with guaranteed available
        # visit twice to have this name bumped to min 2 assignments
        self.visit(node.target)
        # visit the whole function
        self.generic_visit(node)

    def visit_If(self, node: If) -> None:
        # TODO future items: use this together with guaranteed available
        return

    def visit_Name(self, node: Name) -> None:
        if isinstance(node.ctx, Store):
            self.vars[node.id] += 1

    def visit_ClassDef(self, node: ClassDef):
        self.vars[node.name] += 1
        # ignore the content (i.e. attribute names) of class definitions

    def visit_FunctionDef(self, node: FunctionDef):
        self.vars[node.name] += 1
        # visit arguments twice, they are generally assigned more than once
        self.generic_visit(node.args)
        self.generic_visit(node)


class OptimizeConstantFolding(CompilingNodeTransformer):
    step = "Constant folding"

    def __init__(self):
        self.scopes_visible = [
            set(INITIAL_SCOPE.keys()).difference(SAFE_GLOBALS.keys())
        ]
        self.scopes_constants = [dict()]
        self.constants = set()

    def enter_scope(self):
        self.scopes_visible.append(set())
        self.scopes_constants.append(dict())

    def add_var_visible(self, var: str):
        self.scopes_visible[-1].add(var)

    def add_vars_visible(self, var: typing.Iterable[str]):
        self.scopes_visible[-1].update(var)

    def add_constant(self, var: str, value: typing.Any):
        self.scopes_constants[-1][var] = value

    def visible_vars(self):
        res_set = set()
        for s in self.scopes_visible:
            res_set.update(s)
        return res_set

    def constant_vars(self):
        res_d = {}
        for s in self.scopes_constants:
            res_d.update(s)
        return res_d

    def exit_scope(self):
        self.scopes_visible.pop(-1)
        self.scopes_constants.pop(-1)

    def non_overwritten_globals(self):
        overwritten_vars = self.visible_vars()

        def err():
            raise ValueError("Was overwritten!")

        non_overwritten_globals = {
            k: (v if k not in overwritten_vars else err)
            for k, v in SAFE_GLOBALS.items()
        }
        return non_overwritten_globals

    def visit_Module(self, node: Module) -> Module:
        self.enter_scope()
        def_vars_collector = ShallowNameDefCollector()
        def_vars_collector.visit(node)
        def_vars = def_vars_collector.vars
        self.add_vars_visible(def_vars)

        constant_collector = DefinedTimesVisitor()
        constant_collector.visit(node)
        constants = constant_collector.vars
        # if it is only assigned exactly once, it must be a constant (due to immutability)
        self.constants = {c for c, i in constants.items() if i == 1}

        res = self.generic_visit(node)
        self.exit_scope()
        return res

    def visit_FunctionDef(self, node: FunctionDef) -> FunctionDef:
        self.add_var_visible(node.name)
        self.enter_scope()
        self.add_vars_visible(arg.arg for arg in node.args.args)
        def_vars_collector = ShallowNameDefCollector()
        for s in node.body:
            def_vars_collector.visit(s)
        def_vars = def_vars_collector.vars
        self.add_vars_visible(def_vars)

        if node.name in self.constants:
            g = self.non_overwritten_globals()
            l = self.constant_vars()
            exec(unparse(node), g, l)
            # the function is defined and added to the globals
            self.add_constant(node.name, l[node.name])

        res_node = self.generic_visit(node)
        self.exit_scope()
        return res_node

    def visit_ClassDef(self, node: ClassDef):
        if node.name in self.constants:
            g = self.non_overwritten_globals()
            l = self.constant_vars()
            exec(unparse(node), g, l)
            # the class is defined and added to the globals
            self.add_constant(node.name, l[node.name])
        return node

    def visit_Import(self, node: Import):
        if all(n in self.constants for n in node.names):
            g = self.non_overwritten_globals()
            l = self.constant_vars()
            g.update(l)
            new_l = {}
            try:
                exec(unparse(node), g, new_l)
            except:
                pass
            else:
                # the class is defined and added to the globals
                self.scopes_constants[-1].update(new_l)

    def visit_Assign(self, node: Assign):
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if not isinstance(target, Name):
            return node

        if target.id in self.constants:
            g = self.non_overwritten_globals()
            l = self.constant_vars()
            try:
                exec(unparse(node), g, l)
            except:
                pass
            else:
                # the class is defined and added to the globals
                self.add_constant(target.id, l[target.id])
        node.value = self.visit(node.value)
        return node

    def visit_AnnAssign(self, node: AnnAssign):
        target = node.target
        if not isinstance(target, Name):
            return node

        if target.id in self.constants:
            g = self.non_overwritten_globals()
            exec(unparse(node), g, self.constant_vars())
            # the class is defined and added to the globals
            self.add_constant(target.id, g[target.id])
        node.value = self.visit(node.value)
        return node

    def generic_visit(self, node: AST):
        node = super().generic_visit(node)
        if not isinstance(node, expr):
            # only evaluate expressions, not statements
            return node
        if isinstance(node, Constant):
            # prevents unneccessary computations
            return node
        node_source = unparse(node)
        if "print(" in node_source:
            # do not optimize away print statements
            return node
        try:
            # TODO we can add preceding plutusdata definitions here!
            node_eval = eval(
                node_source, self.non_overwritten_globals(), self.constant_vars()
            )
        except Exception as e:
            return node

        if any(isinstance(node_eval, t) for t in ACCEPTED_ATOMIC_TYPES + [list, dict]):
            new_node = Constant(node_eval, None)
            copy_location(new_node, node)
            return new_node
        return node
