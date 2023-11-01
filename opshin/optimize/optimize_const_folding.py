import typing
import logging

from ast import *

from pycardano import PlutusData


try:
    unparse
except NameError:
    from astunparse import unparse

from ..util import (
    CompilingNodeTransformer,
    ShallowNameDefCollector,
    DefinedTimesVisitor,
)
from ..type_inference import INITIAL_SCOPE

"""
Pre-evaluates constant statements
"""

_LOGGER = logging.getLogger(__name__)

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

    def _constant_vars(self):
        res_d = {}
        for s in self.scopes_constants:
            res_d.update(s)
        return res_d

    def exit_scope(self):
        self.scopes_visible.pop(-1)
        self.scopes_constants.pop(-1)

    def _non_overwritten_globals(self):
        overwritten_vars = self.visible_vars()

        def err():
            raise ValueError("Was overwritten!")

        non_overwritten_globals = {
            k: (v if k not in overwritten_vars else err)
            for k, v in SAFE_GLOBALS.items()
        }
        return non_overwritten_globals

    def update_constants(self, node):
        a = self._non_overwritten_globals()
        a.update(self._constant_vars())
        g = a
        l = {}
        try:
            exec(unparse(node), g, l)
        except Exception as e:
            _LOGGER.debug(e)
        else:
            # the class is defined and added to the globals
            self.scopes_constants[-1].update(l)

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
        if node.name in self.constants:
            a = self._non_overwritten_globals()
            a.update(self._constant_vars())
            g = a
            try:
                # we need to pass the global dict as local dict here to make closures possible (rec functions)
                exec(unparse(node), g, g)
            except Exception as e:
                _LOGGER.debug(e)
            else:
                # the class is defined and added to the globals
                self.scopes_constants[-1][node.name] = g[node.name]

        self.enter_scope()
        self.add_vars_visible(arg.arg for arg in node.args.args)
        def_vars_collector = ShallowNameDefCollector()
        for s in node.body:
            def_vars_collector.visit(s)
        def_vars = def_vars_collector.vars
        self.add_vars_visible(def_vars)

        res_node = self.generic_visit(node)
        self.exit_scope()
        return res_node

    def visit_ClassDef(self, node: ClassDef):
        if node.name in self.constants:
            self.update_constants(node)
        return node

    def visit_ImportFrom(self, node: ImportFrom):
        if all(n in self.constants for n in node.names):
            self.update_constants(node)
        return node

    def visit_Import(self, node: Import):
        if all(n in self.constants for n in node.names):
            self.update_constants(node)
        return node

    def visit_Assign(self, node: Assign):
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if not isinstance(target, Name):
            return node

        if target.id in self.constants:
            self.update_constants(node)
        node.value = self.visit(node.value)
        return node

    def visit_AnnAssign(self, node: AnnAssign):
        target = node.target
        if not isinstance(target, Name):
            return node

        if target.id in self.constants:
            self.update_constants(node)
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
            # we add preceding constant plutusdata definitions here!
            g = self._non_overwritten_globals()
            l = self._constant_vars()
            node_eval = eval(node_source, g, l)
        except Exception as e:
            _LOGGER.debug(e)
            return node

        if any(
            isinstance(node_eval, t)
            for t in ACCEPTED_ATOMIC_TYPES + [list, dict, PlutusData]
        ):
            new_node = Constant(node_eval, None)
            copy_location(new_node, node)
            return new_node
        return node
