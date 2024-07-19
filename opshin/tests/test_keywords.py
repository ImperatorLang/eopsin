import unittest

import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from .utils import eval_uplc_value
from . import PLUTUS_VM_PROFILE


hypothesis.settings.load_profile(PLUTUS_VM_PROFILE)


class Keyword_tests(unittest.TestCase):
    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_all_keywords(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z: int) -> int:
    return (x-y)*z

def validator(a: int, b: int, c: int) -> int:
    return simple_example(x=a, y=b, z=c)
"""
        ret = eval_uplc_value(source_code, x, y, z)
        self.assertEqual(ret, (x - y) * z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_mixture_args_and_keywords(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z: int) -> int:
    return (x-y)*z

def validator(a: int, b: int, c: int) -> int:
    return simple_example(a, b, z=c)
"""
        ret = eval_uplc_value(source_code, x, y, z)
        self.assertEqual(ret, (x - y) * z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_keyword_position_independence(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z:int) -> int:
    return (x-y)*z

def validator(a: int, b: int, c: int) -> int:
    return simple_example(z=c, x=a, y=b)
"""
        ret = eval_uplc_value(source_code, x, y, z)
        self.assertEqual(ret, (x - y) * z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_arg_after_keyword_failure(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z:int) -> int:
    return (x-y)*z

def validator(a: int, b: int, c: int) -> int:
    return simple_example(x=a, y=b, c)
"""
        with self.assertRaises(Exception):
            ret = eval_uplc_value(source_code, x, y, z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_too_many_keywords_failure(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int) -> int:
    return x-y

def validator(a: int, b: int, c: int) -> int:
    return simple_example(x=a, y=b, z=c)
"""
        with self.assertRaises(Exception):
            ret = eval_uplc_value(source_code, x, y, z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_incorrect_keywords_failure(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z: int) -> int:
    return (x-y)*z

def validator(a: int, b: int, c: int) -> int:
    return simple_example(x=a, y=b, k=c)
"""
        with self.assertRaises(Exception):
            ret = eval_uplc_value(source_code, x, y, z)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_correct_scope(self, x: int, y: int, z: int):
        source_code = """
def simple_example(x: int, y: int, z: int) -> int:
    def simple_example(new_x: int, new_z: int) -> int:
        return new_x-new_z
    return simple_example(new_x = x, new_z = z) * y

def validator(a: int, b: int, c: int) -> int:
    return simple_example(x=a, y=b, z=c)
"""
        ret = eval_uplc_value(source_code, x, y, z)
        self.assertEqual(ret, (x - z) * y)

    @given(x=st.integers(), y=st.integers(), z=st.integers())
    def test_default(self, x: int, y: int, z: int):
        source_code = f"""
def simple_example(x: int, y: int, z: int={z}) -> int:
    return (x-z)*y

def validator(a: int, b: int) -> int:
    return simple_example(a, b)
"""
        ret = eval_uplc_value(source_code, x, y)
        self.assertEqual(ret, (x - z) * y)

    def test_default_wrong_type(self):
        source_code = f"""
def simple_example(x: int, y: int, z: int="hello") -> int:
    return (x-z)*y

def validator(a: int, b: int) -> int:
    return simple_example(a, b)
"""
        with self.assertRaises(Exception):
            ret = eval_uplc_value(source_code, 1, 2)

    def test_no_allow_validator_default(self):
        source_code = f"""
def validator(a: int, b: int, c:int=1) -> int:
    return a*b*c
"""
        with self.assertRaises(Exception):
            ret = eval_uplc_value(source_code, 1, 2, 2)
