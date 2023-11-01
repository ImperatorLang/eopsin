from ast import *
from copy import copy

from ..util import CompilingNodeVisitor, CompilingNodeTransformer, NameLoadCollector
from ..type_inference import INITIAL_SCOPE
from ..typed_ast import TypedAnnAssign

"""
Removes assignments to variables that are never read
"""


class SafeOperationVisitor(CompilingNodeVisitor):
    step = "Collecting computations that can not throw errors"

    def __init__(self, guaranteed_names):
        self.guaranteed_names = guaranteed_names

    def generic_visit(self, node: AST) -> bool:
        # generally every operation is unsafe except we whitelist it
        return False

    def visit_Lambda(self, node: Lambda) -> bool:
        # lambda definition is fine as it actually doesn't compute anything
        return True

    def visit_Constant(self, node: Constant) -> bool:
        # Constants can not fail
        return True

    def visit_RawPlutoExpr(self, node) -> bool:
        # these expressions are not evaluated further
        return True

    def visit_Name(self, node: Name) -> bool:
        return node.id in self.guaranteed_names


class OptimizeRemoveDeadvars(CompilingNodeTransformer):
    step = "Removing unused variables"

    loaded_vars = None
    # names that are guaranteed to be available to the current node
    # this acts differently to the type inferencer! in particular, ite/while/for all produce their own scope
    guaranteed_avail_names = [
        list(INITIAL_SCOPE.keys()) + ["isinstance", "Union", "Dict", "List"]
    ]

    def guaranteed(self, name: str) -> bool:
        name = name
        for scope in reversed(self.guaranteed_avail_names):
            if name in scope:
                return True
        return False

    def enter_scope(self):
        self.guaranteed_avail_names.append([])

    def exit_scope(self):
        self.guaranteed_avail_names.pop()

    def set_guaranteed(self, name: str):
        self.guaranteed_avail_names[-1].append(name)

    def visit_Module(self, node: Module) -> Module:
        # repeat until no more change due to removal
        # i.e. b = a; c = b needs 2 passes to remove c and b
        node_cp = copy(node)
        self.loaded_vars = None
        while True:
            self.enter_scope()
            # collect all variable names
            collector = NameLoadCollector()
            collector.visit(node_cp)
            loaded_vars = set(collector.loaded.keys()) | {"validator_0"}
            # break if the set of loaded vars did not change -> set of vars to remove does also not change
            if loaded_vars == self.loaded_vars:
                break
            # remove unloaded ones
            self.loaded_vars = loaded_vars
            node_cp.body = [self.visit(s) for s in node_cp.body]
            self.exit_scope()
        return node_cp

    def visit_If(self, node: If):
        node_cp = copy(node)
        node_cp.test = self.visit(node.test)
        self.enter_scope()
        node_cp.body = [self.visit(s) for s in node.body]
        scope_body_cp = self.guaranteed_avail_names[-1].copy()
        self.exit_scope()
        self.enter_scope()
        node_cp.orelse = [self.visit(s) for s in node.orelse]
        scope_orelse_cp = self.guaranteed_avail_names[-1].copy()
        self.exit_scope()
        # what remains after this in the scope is the intersection of both
        for var in set(scope_body_cp).intersection(scope_orelse_cp):
            self.set_guaranteed(var)
        return node_cp

    def visit_While(self, node: While):
        node_cp = copy(node)
        node_cp.test = self.visit(node.test)
        self.enter_scope()
        node_cp.body = [self.visit(s) for s in node.body]
        node_cp.orelse = [self.visit(s) for s in node.orelse]
        self.exit_scope()
        return node_cp

    def visit_For(self, node: For):
        node_cp = copy(node)
        assert isinstance(node.target, Name), "Can only assign to singleton name"
        self.enter_scope()
        self.guaranteed(node.target.id)
        node_cp.body = [self.visit(s) for s in node.body]
        node_cp.orelse = [self.visit(s) for s in node.orelse]
        self.exit_scope()
        return node_cp

    def visit_Assign(self, node: Assign):
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], Name)
            or node.targets[0].id in self.loaded_vars
            or not SafeOperationVisitor(sum(self.guaranteed_avail_names, [])).visit(
                node.value
            )
        ):
            for t in node.targets:
                assert isinstance(
                    t, Name
                ), "Need to have name for dead var remover to work"
                self.set_guaranteed(t.id)
            return self.generic_visit(node)
        return Pass()

    def visit_AnnAssign(self, node: TypedAnnAssign):
        if (
            not isinstance(node.target, Name)
            or node.target.id in self.loaded_vars
            or not SafeOperationVisitor(sum(self.guaranteed_avail_names, [])).visit(
                node.value
            )
            # only upcasts are safe!
            or not node.target.typ >= node.value.typ
        ):
            assert isinstance(
                node.target, Name
            ), "Need to have assignments to name for dead var remover to work"
            self.set_guaranteed(node.target.id)
            return self.generic_visit(node)
        return Pass()

    def visit_ClassDef(self, node: ClassDef):
        if node.name in self.loaded_vars:
            self.set_guaranteed(node.name)
            return node
        return Pass()

    def visit_FunctionDef(self, node: FunctionDef):
        node_cp = copy(node)
        if node.name in self.loaded_vars:
            self.set_guaranteed(node.name)
            self.enter_scope()
            # variable names are available here
            for a in node.args.args:
                self.set_guaranteed(a.arg)
            node_cp.body = [self.visit(s) for s in node.body]
            self.exit_scope()
            return node_cp
        return Pass()
