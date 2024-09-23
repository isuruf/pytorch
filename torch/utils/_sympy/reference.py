# mypy: allow-untyped-defs
import math
import operator
from functools import reduce

import sympy

import torch
from torch.utils._sympy.functions import (
    _keep_float,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToInt,
)


# The sympy interpretation of operators.  It will also sometimes work with
# plain int/float, but if you do certain operations you will get out a
# sympy.Basic in the end.  If you want the Python/FX traceable interpretation,
# check PythonReferenceAnalysis.
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    @staticmethod
    def constant(c, dtype):
        return sympy.sympify(c)

    @staticmethod
    def or_(*args):
        return reduce(operator.or_, args)

    @staticmethod
    def and_(*args):
        return reduce(operator.and_, args)

    @staticmethod
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        return a < b

    @staticmethod
    def gt(a, b):
        return a > b

    @staticmethod
    def le(a, b):
        return a <= b

    @staticmethod
    def ge(a, b):
        return a >= b

    @staticmethod
    def not_(a):
        assert not isinstance(a, bool)
        return ~a

    @staticmethod
    def reciprocal(x):
        return FloatTrueDiv(1.0, x)

    @staticmethod
    def square(x):
        return PowByNatural(x, 2)

    @staticmethod
    def trunc_to_int(x, dtype):
        return TruncToInt(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return sympy.ceiling(x)

    @staticmethod
    def floor_to_int(x, dtype):
        return sympy.floor(x)

    @staticmethod
    def floor(x):
        return _keep_float(sympy.floor)(x)

    @staticmethod
    def ceil(x):
        return _keep_float(sympy.ceiling)(x)

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return ToFloat(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def mod(x, y):
        return Mod(x, y)

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return FloatTrueDiv(a, b)

    @staticmethod
    def int_truediv(a, b):
        return IntTrueDiv(a, b)

    @staticmethod
    def floordiv(a, b):
        return FloorDiv(a, b)

    @staticmethod
    def truncdiv(a, b):
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    @_keep_float
    def add(*args):
        return ReferenceAnalysis.add_or_mul(args, sympy.Add, operator.add)

    @staticmethod
    @_keep_float
    def mul(*args):
        return ReferenceAnalysis.add_or_mul(args, sympy.Mul, operator.mul)

    @staticmethod
    def add_or_mul(args, sympy_cls, op):
        syms, others = [], []
        for arg in args:
            if isinstance(arg, sympy.Basic):
                syms.append(arg)
            else:
                others.append(arg)
        if syms:
            return reduce(op, others) + sympy_cls(*syms)
        else:
            return reduce(op, others)


    @staticmethod
    def sub(a, b):
        return _keep_float(operator.sub)(a, b)

    @staticmethod
    def exp(x):
        return OpaqueUnaryFn_exp(x)

    @staticmethod
    def log(x):
        return OpaqueUnaryFn_log(x)

    @staticmethod
    def sqrt(x):
        return OpaqueUnaryFn_sqrt(x)

    @staticmethod
    def pow(a, b):
        return _keep_float(FloatPow)(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        return PowByNatural(a, b)

    @staticmethod
    def minimum(a, b):
        return Min(a, b)

    @staticmethod
    def maximum(a, b):
        return Max(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        return RoundToInt(a)

    @staticmethod
    def round_decimal(a, b):
        return RoundDecimal(a, b)


# Unlike ReferenceAnalysis, does NOT sympyify, instead, works with plain
# Python types and is FX traceable.  Inheritance here is purely for code
# sharing (TODO: considering splitting out a BaseReferenceAnalysis).
class PythonReferenceAnalysis(ReferenceAnalysis):
    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def mod(x, y):
        return x % y

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return torch.sym_float(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def exp(x):
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x):
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def sqrt(x):
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    @staticmethod
    def minimum(*args):
        return reduce(torch.sym_min, args)

    @staticmethod
    def maximum(args):
        return reduce(torch.sym_min, args)

    @staticmethod
    def floor_to_int(x, dtype):
        return math.floor(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return math.ceil(x)

    @staticmethod
    def floor(x):
        return float(math.floor(x))

    @staticmethod
    def ceil(x):
        return float(math.ceil(x))

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def pow_by_natural(a, b):
        # Pray that safe_pow is not needed here lol.  In particular, this
        # never participates in VR low/high ranges, so overflow should be
        # unlikely
        return a**b

    @staticmethod
    def round_to_int(a, dtype):
        return round(a)

    @staticmethod
    def round_decimal(a, b):
        return round(a, ndigits=b)
