from typing import Optional, Union

from ast import *
from copy import copy
from collections import defaultdict

from ..util import (
    CompilingNodeVisitor,
    CompilingNodeTransformer,
    DefinedTimesVisitor,
    NameLoadCollector,
)
from ..type_inference import INITIAL_SCOPE
from ..typed_ast import TypedAnnAssign

"""
Inline constants that are only assigned once and only read once
"""


class OptimizeInlineConstvars(CompilingNodeTransformer):
    step = "Inline variables that are constant (i.e. written only once)"

    constants = set()
    constant_value = defaultdict(list)
    loaded_vars = set()

    def visit_Module(self, node: Module) -> Module:
        # repeat until no more change due to removal
        # i.e. b = a; c = b needs 2 passes to remove c and b
        node_cp = copy(node)
        # collect all variable names
        constant_collector = DefinedTimesVisitor()
        constant_collector.visit(node)
        constants = constant_collector.vars
        # if it is only assigned exactly once, it must be a constant (due to immutability)
        loaded_collector = NameLoadCollector()
        loaded_collector.visit(node)
        loaded = loaded_collector.loaded
        # if it is only read exactly once, there is no downside to inlining it (i.e. no multiple computations)
        self.constants = {c for c, i in constants.items() if i == 1 and loaded[c] == 1}
        node_cp.body = [self.visit(s) for s in node_cp.body]
        return node_cp

    def visit_Assign(self, node: Assign) -> Optional[Assign]:
        assert len(node.targets) == 1
        t = node.targets[0]
        if (
            isinstance(t, Name)
            and t.id in self.constants
            and t.id not in self.loaded_vars
        ):
            # we may only replace those loaded AFTER they are assigned
            # (otherwise a NameError could be thrown), so we will just add them here
            self.constant_value[t.id].append(self.visit(node.value))
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: Name) -> Union[Name, expr]:
        if isinstance(node.ctx, Load):
            # mark the variable as loaded
            self.loaded_vars.add(node.id)
            if node.id in self.constant_value:
                # Only replace those where we have a value (i.e. do not replace functions or class definitions)
                return self.constant_value[node.id][0]
        return self.generic_visit(node)
