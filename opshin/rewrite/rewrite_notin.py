from ast import *
from copy import copy

from ..util import CompilingNodeTransformer

"""
Rewrites all occurences of (a not in b) into (not (a in b)).
"""


class RewriteNotIn(CompilingNodeTransformer):
    step = "Rewriting (not in)"

    def visit_Compare(self, node: Compare) -> AST:
        assert (
            len(node.ops) == 1
        ), "RewriteNotIn only works on single comparisons, need to run RewriteComparisonChaining first"
        if not isinstance(node.ops[0], NotIn):
            return self.generic_visit(node)
        new_node = UnaryOp(Not(), Compare(node.left, [In()], node.comparators))
        return self.generic_visit(new_node)
