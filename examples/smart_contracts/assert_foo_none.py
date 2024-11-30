#!opshin
from opshin.prelude import *


def foo() -> None:
    return None


def validator(datum: int, redeemer: int, context: ScriptContext) -> None:
    assert foo(), f"Always fails!"
