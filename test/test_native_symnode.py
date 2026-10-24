# Owner(s): ["module: dynamic shapes"]
import contextlib
import copy
import enum
import inspect
import operator
import os
import pickle
import random
import re
import subprocess
import sys
import types
from unittest import mock

import sympy

import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental import proxy_tensor, sym_node, symbolic_shapes
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.sym_node import _NO_HINT, SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Min,
    Mod,
    ModularIndexing,
    PowByNatural,
    PythonMod,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


NativeUnsupported = torch._C._symbolic.NativeUnsupported
_NativeSymNode = torch._C._symbolic._NativeSymNode

s0 = sympy.Symbol("s0", integer=True, positive=True)
s1 = sympy.Symbol("s1", integer=True, positive=True)


class TestNativeSymNodeFlag(TestCase):
    def _read_flag(self, env_value):
        env = dict(os.environ)
        env.pop("CPP_SYMNODE", None)
        if env_value is not None:
            env["CPP_SYMNODE"] = env_value
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch; print(torch._dynamo.config.use_cpp_symnode)",
            ],
            env=env,
            text=True,
        )
        return out.strip().splitlines()[-1]

    def test_flag_env(self):
        self.assertEqual(self._read_flag(None), "False")
        self.assertEqual(self._read_flag("0"), "False")
        self.assertEqual(self._read_flag("1"), "True")


def symbol_samples(s, rng):
    """A value of s that its assumptions allow."""
    if s.is_positive:
        return rng.randint(1, 40)
    if s.is_nonnegative:
        return rng.randint(0, 40)
    return rng.randint(-40, 40)


def assert_same_function(test, got, want, points=20):
    """got and want agree wherever both are defined, at sampled symbol values."""
    if got == want:
        return
    syms = sorted(got.free_symbols | want.free_symbols, key=str)
    # Symbol is a Boolean too, so ask for the boolean kinds.
    bool_kinds = (
        sympy.logic.boolalg.BooleanAtom,
        sympy.logic.boolalg.BooleanFunction,
        sympy.core.relational.Relational,
    )
    test.assertEqual(
        isinstance(got, bool_kinds), isinstance(want, bool_kinds), f"{got} vs {want}"
    )
    rng = random.Random(0)
    for _ in range(points):
        point = {s: symbol_samples(s, rng) for s in syms}
        try:
            w = value_at(want, point)
            g = value_at(got, point)
        except (ZeroDivisionError, OverflowError):
            continue
        test.assertEqual(g, w, f"{got} vs {want} at {point}")


def sample_points(rng, syms, n=20, lo=1, hi=40):
    return [{s: rng.randint(lo, hi) for s in syms} for _ in range(n)]


def value_at(e, point):
    """e at point, as a Python int or bool."""
    r = e.xreplace({s: sympy.Integer(v) for s, v in point.items()})
    if r in (sympy.true, sympy.false):
        return bool(r)
    return int(r)


def random_int_expr(rng, depth, leaves):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(leaves)
    a = random_int_expr(rng, depth - 1, leaves)
    b = random_int_expr(rng, depth - 1, leaves)
    op = rng.choice(["add", "sub", "mul", "floordiv", "pymod", "max", "min"])
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    # A divisor that is positive at every sample point.
    d = rng.choice([sympy.Integer(rng.randint(1, 6)), s1, s1 + 2])
    if op == "floordiv":
        return FloorDiv(a, d)
    if op == "pymod":
        return PythonMod(a, d)
    return (Max if op == "max" else Min)(a, b)


INT_LEAVES = [s0, s1, *map(sympy.Integer, range(-3, 4))]


class TestNativeExpr(TestCase):
    def setUp(self):
        super().setUp()
        self.arena = torch._C._symbolic._Arena()

    def assertSameValues(self, want, got, syms=(s0, s1), seed=0):
        rng = random.Random(seed)
        for point in sample_points(rng, syms, hi=8):
            self.assertEqual(
                value_at(want, point), value_at(got, point), f"{want} vs {got}"
            )

    @parametrize(
        "e",
        [
            s0,
            sympy.Integer(-7),
            sympy.Rational(3, 4) * s0,
            2 * s0 + 3,
            s0 * s1 - 4,
            s0**3,
            FloorDiv(s0 + 3, 4),
            CleanDiv(4 * s0, 4),
            Mod(s0, 3),
            PythonMod(s0 - s1, 5),
            PowByNatural(s0, s1),
            Max(s0, s1 + 1),
            Min(s0, 3),
            IsNonOverlappingAndDenseIndicator(s0, s1, s1, 1),
            sympy.Eq(s0, 3),
            sympy.Ne(s0, s1),
            sympy.Lt(s0, s1),
            sympy.Ge(2 * s0, s1 + 1),
            sympy.And(sympy.Gt(s0, 2), sympy.Le(s1, 7)),
            sympy.Or(sympy.Eq(s0, 1), sympy.Not(sympy.Eq(s1, 2))),
        ],
    )
    def test_roundtrip(self, e):
        r = self.arena.roundtrip(e)
        self.assertIsInstance(r, sympy.Basic)
        if isinstance(e, sympy.Expr) and e.is_number:
            self.assertEqual(r, e)
        else:
            self.assertSameValues(e, r)

    def test_roundtrip_keeps_symbols(self):
        self.assertIs(self.arena.roundtrip(s0), s0)
        self.assertIs(self.arena.roundtrip(s0 + s1).args[0].func, sympy.Symbol)

    def test_big_integers(self):
        # Bounds outside int64 come back as exact Python ints.
        big = 2**40
        ranges = {s0: (big, big), s1: (big, big + 3)}
        lower, upper = self.arena.bound_sympy(-s0 * s1, ranges)
        self.assertEqual((lower, upper), (-big * (big + 3), -big * big))
        lower, upper = self.arena.bound_sympy(s0 * s1 * s1, ranges)
        self.assertEqual((lower, upper), (big**3, big * (big + 3) ** 2))

    def test_infinity(self):
        self.assertEqual(self.arena.roundtrip(int_oo), int_oo)
        self.assertEqual(self.arena.roundtrip(-int_oo), -int_oo)

    @parametrize(
        "e",
        [
            sympy.Float(1.5),
            sympy.Symbol("x", real=True) + sympy.Float(2.0),
            ModularIndexing(s0, 2, 3),
        ],
    )
    def test_unsupported(self, e):
        with self.assertRaises(NativeUnsupported):
            self.arena.roundtrip(e)

    def test_symbol_assumption_conflict(self):
        self.arena.roundtrip(sympy.Symbol("q", integer=True, positive=True))
        with self.assertRaises(NativeUnsupported):
            self.arena.roundtrip(sympy.Symbol("q", integer=True))

    @parametrize("name", ["FloorDiv", "Mod", "PythonMod", "PowByNatural"])
    def test_function_numbers(self, name):
        cls = getattr(torch.utils._sympy.functions, name)
        rng = random.Random(1)
        for _ in range(200):
            if name == "PowByNatural":
                a, b = rng.randint(-5, 5), rng.randint(0, 6)
            elif name == "Mod":
                a, b = rng.randint(0, 50), rng.randint(1, 9)
            else:
                a, b = rng.randint(-50, 50), rng.choice([-7, -2, -1, 1, 3, 8])
            args = [sympy.Integer(a), sympy.Integer(b)]
            self.assertEqual(self.arena.function(name, args), cls(*args), str(args))

    @parametrize("name", ["FloorDiv", "PythonMod"])
    def test_function_symbolic(self, name):
        cls = getattr(torch.utils._sympy.functions, name)
        rng = random.Random(2)
        for _ in range(100):
            base = random_int_expr(rng, 2, INT_LEAVES)
            divisor = rng.choice([sympy.Integer(rng.randint(1, 12)), s1, 2 * s1])
            got = self.arena.function(name, [base, divisor])
            self.assertSameValues(cls(base, divisor), got)

    def test_function_folding(self):
        a = self.arena
        self.assertEqual(a.function("FloorDiv", [4 * s0 + 6, sympy.Integer(4)]), s0 + 1)
        self.assertEqual(
            a.function("FloorDiv", [FloorDiv(s0, 2), sympy.Integer(3)]),
            FloorDiv(s0, 6),
        )
        self.assertEqual(a.function("Mod", [6 * s0, sympy.Integer(3)]), 0)
        self.assertEqual(
            a.function("PythonMod", [4 * s0 + s1, sympy.Integer(4)]),
            PythonMod(s1, 4),
        )
        self.assertEqual(a.function("PowByNatural", [s0, sympy.Integer(2)]), s0**2)

    @parametrize("name", ["FloorDiv", "Mod", "PythonMod"])
    def test_zero_divisor(self, name):
        with self.assertRaises(NativeUnsupported):
            self.arena.function(name, [s0, sympy.Integer(0)])

    def test_bound_sympy_sound(self):
        ranges = {s0: (2, int_oo), s1: (2, 9)}
        rng = random.Random(3)
        checked = 0
        for _ in range(300):
            e = random_int_expr(rng, 3, INT_LEAVES)
            if rng.random() < 0.3:
                e = rng.choice([sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne])(
                    e, random_int_expr(rng, 2, INT_LEAVES)
                )
            try:
                lower, upper = self.arena.bound_sympy(e, ranges)
            except NativeUnsupported:
                continue
            checked += 1
            for point in sample_points(rng, [s0], n=10, lo=2, hi=60):
                point[s1] = rng.randint(2, 9)
                v = value_at(e, point)
                if isinstance(v, bool):
                    # [true, true] rules out False, [false, false] rules out True.
                    self.assertFalse(lower == sympy.true and not v, str(e))
                    self.assertFalse(upper == sympy.false and v, str(e))
                else:
                    self.assertTrue(
                        lower <= v <= upper,
                        f"{e} = {v} at {point} not in [{lower}, {upper}]",
                    )
        self.assertGreater(checked, 200)

    @parametrize(
        "e,want",
        [
            (s0 * s1 - 3, (1, int_oo)),
            (FloorDiv(s0, s1), (0, int_oo)),
            (PythonMod(s0, s1), (0, 8)),
            (Max(s1, 3), (3, 9)),
            (s1**2, (4, 81)),
            (sympy.Gt(s0, 1), (sympy.true, sympy.true)),
            (sympy.Gt(s0, 5), (sympy.false, sympy.true)),
            (sympy.Le(s0 + s1, s0 + 9), (sympy.true, sympy.true)),
            (sympy.Eq(s1, 10), (sympy.false, sympy.false)),
        ],
    )
    def test_bound_sympy(self, e, want):
        got = self.arena.bound_sympy(e, {s0: (2, int_oo), s1: (2, 9)})
        self.assertEqual(got, tuple(sympy.sympify(w) for w in want))


class TestNativeShapeEnv(TestCase):
    # (hint, lower, upper) of the symbols s0..s3.
    SPECS = [(3, 2, int_oo), (5, 2, int_oo), (7, 4, 64), (9, 0, int_oo)]

    def make_env(self):
        env = ShapeEnv(_allow_native=True)
        syms = [
            env.create_symbol(hint, ConstantSource(f"x{i}"), DimDynamic.DYNAMIC)
            for i, (hint, _, _) in enumerate(self.SPECS)
        ]
        for s, (_, lo, hi) in zip(syms, self.SPECS):
            env.var_to_range[s] = ValueRanges(lo, hi)
        native = torch._C._symbolic.NativeShapeEnv(torch._C._symbolic._Arena())
        for s, (hint, lo, hi) in zip(syms, self.SPECS):
            native.mirror_symbol(s, hint, sympy.Integer(lo), hi)
        return env, native, syms

    def points(self, syms, rng):
        for _ in range(30):
            yield {
                s: rng.randint(lo, min(hi, lo + 50) if hi is not int_oo else lo + 50)
                for s, (_, lo, hi) in zip(syms, self.SPECS)
            }

    def test_mirrored(self):
        _, native, syms = self.make_env()
        self.assertEqual(native.mirrored(syms[2]), (7, 4, 64, False))
        self.assertIsNone(native.mirrored(sympy.Symbol("zz", integer=True)))

    def test_static_eval_sound(self):
        rng = random.Random(4)
        env, native, syms = self.make_env()
        leaves = [*syms, *map(sympy.Integer, range(-2, 5))]
        answered = 0
        for _ in range(300):
            a = random_int_expr(rng, 2, leaves)
            b = random_int_expr(rng, 2, leaves)
            e = rng.choice(
                [sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne, sympy.Gt, sympy.Ge]
            )(a, b)
            if e in (sympy.true, sympy.false):
                continue
            try:
                ok, r = native.static_eval(e)
            except NativeUnsupported:
                continue
            if not ok or r is None:
                continue
            answered += 1
            for point in self.points(syms, rng):
                self.assertEqual(value_at(e, point), bool(r), f"{e} at {point}")
            want = env._maybe_evaluate_static(e)
            if want is not None:
                self.assertEqual(want, r, str(e))
        self.assertGreater(answered, 10)

    def test_simplify_min_max(self):
        _, native, syms = self.make_env()
        s, t, u, _ = syms
        self.assertEqual(native.simplify(Max(s, s + 1)), s + 1)
        self.assertEqual(native.simplify(Min(u, 3)), 3)
        self.assertEqual(native.simplify(Max(s, t)), Max(s, t))

    def test_evaluate_expr(self):
        _, native, syms = self.make_env()
        s, t, u, _ = syms
        self.assertEqual(native.evaluate_expr(sympy.Gt(s, 1)), sympy.true)
        self.assertEqual(native.evaluate_expr(sympy.Ge(u, 4)), sympy.true)
        self.assertEqual(native.evaluate_expr(sympy.Gt(u, 64)), sympy.false)
        self.assertIsNone(native.evaluate_expr(sympy.Lt(s, t), True))
        self.assertEqual(native.evaluate_expr(sympy.Integer(3), 3), 3)
        self.assertIsNone(native.evaluate_expr(sympy.Integer(3), 4))

    def test_not_pristine(self):
        _, native, syms = self.make_env()
        e = sympy.Gt(syms[0], 1)
        self.assertEqual(native.static_eval(e), (True, sympy.true))
        native.mark_not_pristine()
        self.assertEqual(native.static_eval(e), (False, None))
        self.assertIsNone(native.evaluate_expr(e))


class TestNativeShapeEnvSync(TestCase):
    # (name, mutation of a ShapeEnv with symbols s, t, clears replacements_empty)
    MUTATIONS = [
        ("guard", lambda env, s, t: env.evaluate_expr(sympy.Eq(s, 5)), True),
        (
            "guard_lt",
            lambda env, s, t: env.evaluate_expr(sympy.Lt(s, t), hint=True),
            False,
        ),
        ("guards_append", lambda env, s, t: env.guards.append(None), False),
        ("axioms", lambda env, s, t: env.axioms.update({s > 1: sympy.true}), False),
        ("replacement", lambda env, s, t: env._set_replacement(s, t, "test"), True),
        ("replacements_pop", lambda env, s, t: env.replacements.pop(s, None), True),
        (
            "range",
            lambda env, s, t: env._update_var_to_range(s, ValueRanges(3, 8)),
            False,
        ),
        (
            "range_setitem",
            lambda env, s, t: env.var_to_range.__setitem__(s, ValueRanges(2, 9)),
            False,
        ),
        ("range_del", lambda env, s, t: env.var_to_range.pop(t), False),
        ("divisible", lambda env, s, t: env._add_divisible(Mod(s, 2)), False),
        ("size_like", lambda env, s, t: env._constrain_range_for_size(s), False),
        (
            "runtime_assert_setdefault",
            lambda env, s, t: env.deferred_runtime_asserts.setdefault(None, []),
            False,
        ),
        (
            "backed_var_to_val",
            lambda env, s, t: env.add_backed_var_to_val(
                sympy.Symbol("x", integer=True), 4
            ),
            False,
        ),
    ]
    DISABLING_CONFIGS = [
        ("backed_size_oblivious", True),
        ("aggressive_guard_free_semantics", 1),
        ("symbol_guard_limit_before_specialize", 3),
        ("translation_validation", True),
    ]

    # Queries over the symbols s = 5, t = 7 of make_env.
    QUERY_POOL = [
        lambda s, t: sympy.Gt(s, 1),
        lambda s, t: sympy.Lt(s, t),
        lambda s, t: sympy.Eq(s, 5),
        lambda s, t: sympy.Ge(s + t, 4),
        lambda s, t: sympy.Ne(s * t, 0),
        lambda s, t: sympy.Le(s, 2 * t),
        lambda s, t: sympy.Gt(t, 3),
        lambda s, t: sympy.Eq(Mod(s, 2), 1),
        lambda s, t: sympy.Lt(FloorDiv(t, 2), s),
        lambda s, t: sympy.Eq(t - s, 2),
        lambda s, t: sympy.Le(Max(s, t), s + t),
    ]

    def make_env(self, allow_native=True, **kwargs):
        env = ShapeEnv(_allow_native=allow_native, **kwargs)
        s = env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC)
        t = env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC)
        return env, s, t

    @staticmethod
    def answer(env, evaluate, e, hint=None, fallback_value=None):
        """A query as a native SymNode would issue it: native first, then Python."""
        native = env._native_env
        n = None
        if native is not None:
            try:
                native.arena.roundtrip(e)
                n = e
            except NativeUnsupported:
                pass
        if evaluate:
            if n is not None:
                r = native.evaluate_expr(n, hint, fallback_value)
                if r is not None:
                    return r
            # fx_node=False, as SymNode passes it.
            return env.evaluate_expr(e, hint, False, False, fallback_value)
        if n is not None:
            answered, r = native.static_eval(n)
            if answered:
                return r
        try:
            return env._maybe_evaluate_static(e)
        except Exception:
            return None

    def run_steps(self, allow_native, steps):
        env, s, t = self.make_env(allow_native)
        answers = []
        for evaluate, i, use_hint, suppress in steps:
            e = self.QUERY_POOL[i](s, t)
            hint = bool(e.xreplace({s: 5, t: 7})) if use_hint else None
            with env.suppress_guards() if suppress else contextlib.nullcontext():
                answers.append(self.answer(env, evaluate, e, hint))
        return env, answers

    def test_flag(self):
        self.assertIsNone(ShapeEnv(_allow_native=False)._native_env)
        for on in (False, True):
            with torch._dynamo.config.patch(use_cpp_symnode=on):
                env = ShapeEnv()
            self.assertEqual(env._native_env is not None, on)

    def test_flag_off_state(self):
        env = ShapeEnv(_allow_native=False)
        for name in (
            "guards",
            "axioms",
            "replacements",
            "var_to_range",
            "divisible",
            "size_like",
            "deferred_runtime_asserts",
        ):
            self.assertIn(type(getattr(env, name)), (list, dict, set), name)

    @parametrize("name,value", DISABLING_CONFIGS)
    def test_disabling_config(self, name, value):
        with symbolic_shapes.config.patch(**{name: value}):
            self.assertIsNone(ShapeEnv(_allow_native=True)._native_env)

    def test_disabling_settings(self):
        for kwargs in (
            {"prefer_deferred_runtime_asserts_over_guards": True},
            {"trace_asserts": True},
        ):
            self.assertIsNone(ShapeEnv(_allow_native=True, **kwargs)._native_env)
        self.assertIsNone(
            ShapeEnv(_allow_native=True, should_record_events=True)._native_env
        )

    def test_mirror(self):
        env, s, t = self.make_env()
        native = env._native_env
        # Duck sizing reuses d; positive=None gives an unbounded range.
        d = env.create_symbol(11, ConstantSource("c"))
        self.assertEqual(env.create_symbol(11, ConstantSource("c2")), d)
        u = env.create_symbol(9, ConstantSource("d"), DimDynamic.DYNAMIC, positive=None)
        f = env.create_symbol(2.5, ConstantSource("e"), DimDynamic.DYNAMIC)
        big = env.create_symbol(2**70, ConstantSource("g"), DimDynamic.DYNAMIC)
        self.assertEqual(native.mirrored(s), (5, 2, int_oo, False))
        self.assertEqual(native.mirrored(t), (7, 2, int_oo, False))
        self.assertEqual(native.mirrored(u), (9, -int_oo, int_oo, False))
        self.assertIsNone(native.mirrored(f))
        self.assertIsNone(native.mirrored(big))
        self.assertEqual(native.mirrored(d), (11, 2, int_oo, False))
        for sym in (s, t, u, d):
            vr = env.var_to_range[sym]
            self.assertEqual(
                native.mirrored(sym),
                (
                    int(env.backed_var_to_val[sym]),
                    vr.lower,
                    vr.upper,
                    sym in env.size_like,
                ),
            )
        # Unbacked symbols are never mirrored and creating them keeps the env pristine.
        i0 = env.create_unbacked_symint().node.expr
        self.assertIsNone(native.mirrored(i0))
        self.assertTrue(native.pristine)
        # Static answers do not guard.
        self.assertEqual(env.evaluate_expr(sympy.Gt(s, 1)), sympy.true)
        self.assertTrue(native.pristine)

    def test_constraint_range(self):
        env = ShapeEnv(_allow_native=True)
        vr = ValueRanges(3, 10)
        constraint = symbolic_shapes.StrictMinMaxConstraint(vr=vr, warn_only=False)
        s = env.create_symbol(
            5, ConstantSource("a"), DimDynamic.DYNAMIC, constraint_dim=constraint
        )
        self.assertEqual(env._native_env.mirrored(s), (5, 3, 10, False))
        self.assertFalse(env._native_env.pristine)

    @parametrize(
        "name,mutate,clears_replacements", MUTATIONS, name_fn=lambda name, *_: name
    )
    def test_mutation(self, name, mutate, clears_replacements):
        env, s, t = self.make_env()
        native = env._native_env
        self.assertTrue(native.pristine)
        mutate(env, s, t)
        self.assertFalse(native.pristine)
        self.assertEqual(native.replacements_empty, not clears_replacements)

    def test_notify_before(self):
        env, s, t = self.make_env()
        seen = []
        env.guards._on_mutate = lambda: seen.append(list(env.guards))
        env.guards.append(1)
        env.guards += [2]
        self.assertEqual(seen, [[], [1]])
        self.assertEqual(env.guards, [1, 2])

    def test_update_divisible(self):
        env, s, t = self.make_env()
        env._update_divisible()
        self.assertTrue(env._native_env.pristine)
        env.divisible.add(Mod(s, 2))
        self.assertFalse(env._native_env.pristine)

    def test_copy_and_pickle(self):
        import copy
        import pickle

        env, s, t = self.make_env()
        env.evaluate_expr(sympy.Eq(s, 5))
        env2 = copy.deepcopy(env)
        self.assertIsNone(env2._native_env)
        self.assertIs(type(env2.guards), list)
        self.assertIs(type(env2.replacements), dict)
        self.assertEqual(env2.replacements, env.replacements)
        self.assertIsNone(pickle.loads(pickle.dumps(env._native_env)))
        for name in (
            "axioms",
            "replacements",
            "var_to_range",
            "size_like",
            "_symop_cache",
        ):
            value = getattr(env, name)
            self.assertIs(
                type(pickle.loads(pickle.dumps(value))), type(value).__mro__[-2]
            )
        from torch.fx._graph_pickler import _ShapeEnvPickleData

        self.assertNotIn("_native_env", _ShapeEnvPickleData(env).data)

    def test_check_equal(self):
        def run(allow_native):
            env = ShapeEnv(_allow_native=allow_native)
            s = env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC)
            t = env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC)
            env.evaluate_expr(sympy.Lt(s, t))
            env.evaluate_expr(sympy.Eq(s * 2, 10))
            env._constrain_range_for_size(t)
            return env

        on, off = run(True), run(False)
        self.assertIsNotNone(on._native_env)
        on.check_equal(off)

    def test_take_queries(self):
        env, s, t = self.make_env()
        native = env._native_env
        a = native.arena
        gt, ge = sympy.Gt(s, 1), sympy.Ge(s + t, 4)
        lt = sympy.Lt(s, t)
        self.assertEqual(native.static_eval(gt)[0], True)
        self.assertEqual(native.static_eval(lt), (True, None))
        self.assertIsNotNone(native.evaluate_expr(ge, True, False))
        with env.suppress_guards():
            self.assertIsNotNone(native.evaluate_expr(ge, True, False))
        self.assertIsNotNone(native.evaluate_expr(ge, 1))
        self.assertEqual(native.static_eval(gt)[0], True)
        # Unanswered queries are not logged.
        self.assertIsNone(native.evaluate_expr(lt))
        self.assertTrue(torch._C._symbolic._native_queries_pending())
        queries = native.take_queries()
        self.assertEqual([q[0] for q in queries], sorted(q[0] for q in queries))
        self.assertEqual(
            [q[1:] for q in queries],
            [
                (a.roundtrip(lt), False, None, None, False, None),
                (a.roundtrip(ge), True, True, False, False, sympy.true),
                (a.roundtrip(ge), True, True, False, True, sympy.true),
                (a.roundtrip(ge), True, 1, None, False, sympy.true),
                (a.roundtrip(gt), False, None, None, False, sympy.true),
            ],
        )
        self.assertEqual(native.take_queries(), [])

    def test_static_memo(self):
        env, s, t = self.make_env()
        native = env._native_env
        e = native.arena.roundtrip(sympy.Ge(2 * s * t + s, s + 4))
        n = e
        # Memoized answers are still logged for replay.
        for _ in range(2):
            answered, got = native.static_eval(n)
            self.assertTrue(answered)
            self.assertEqual(got, sympy.true)
            self.assertEqual([q[1] for q in native.take_queries()], [e])
        env.guards.append(None)
        self.assertFalse(native.pristine)
        self.assertEqual(native.static_eval(n), (False, None))
        with self.assertRaises(NativeUnsupported):
            native.maybe_evaluate_static(n)

    def test_flush_populates_python_caches(self):
        env, s, t = self.make_env()
        native = env._native_env
        static_cache = ShapeEnv._maybe_evaluate_static
        evaluate_cache = ShapeEnv._inner_evaluate_expr
        lt, ge = sympy.Lt(s, t), sympy.Ge(s + t, 4)
        self.assertEqual(native.static_eval(lt), (True, None))
        with env.suppress_guards():
            self.assertIsNotNone(native.evaluate_expr(ge, None, True))
        self.assertTrue(torch._C._symbolic._native_queries_pending())
        # Any Python evaluation flushes first.
        env.evaluate_expr(sympy.Gt(s, 1))
        self.assertEqual(native.take_queries(), [])
        self.assertFalse(ShapeEnv._suppress_guards_tls())
        self.assertFalse(torch._C._symbolic._suppress_guards())
        hits = static_cache.cache_info().hits
        self.assertIsNone(env._maybe_evaluate_static(lt))
        self.assertEqual(static_cache.cache_info().hits, hits + 1)
        hits = evaluate_cache.cache_info().hits
        with env.suppress_guards():
            env.evaluate_expr(ge, None, False, fallback_value=True)
        self.assertEqual(evaluate_cache.cache_info().hits, hits + 1)
        self.assertTrue(native.pristine)

    def test_flush_on_mutation(self):
        env, s, t = self.make_env()
        native = env._native_env
        order = []
        native.static_eval(sympy.Gt(s, 1))
        with mock.patch.object(
            env,
            "_maybe_evaluate_static",
            side_effect=lambda e: order.append(("replay", len(env.guards))),
        ):
            env.guards.append(None)
        self.assertEqual(order, [("replay", 0)])
        self.assertFalse(native.pristine)

    def test_flush_on_cache_hit(self):
        env, s, t = self.make_env()
        native = env._native_env
        env._maybe_evaluate_static(sympy.Gt(s, 1))
        native.static_eval(sympy.Lt(s, t))
        hits = ShapeEnv._maybe_evaluate_static.cache_info().hits
        env._maybe_evaluate_static(sympy.Gt(s, 1))
        self.assertEqual(native.take_queries(), [])
        # The replayed Lt clears nothing; Gt stays a hit.
        self.assertEqual(ShapeEnv._maybe_evaluate_static.cache_info().hits, hits + 1)

    def test_flush_order_across_envs(self):
        env1, s1_, _ = self.make_env()
        env2, s2_, _ = self.make_env()
        order = []

        def replay(env, name):
            return lambda e: order.append((name, e))

        for env, s, name in ((env1, s1_, "a"), (env2, s2_, "b"), (env1, s1_, "c")):
            e = sympy.Gt(s, 1) if name != "c" else sympy.Gt(s, 0)
            env._native_env.static_eval(e)
        with (
            mock.patch.object(
                env1, "_maybe_evaluate_static", side_effect=replay(env1, 1)
            ),
            mock.patch.object(
                env2, "_maybe_evaluate_static", side_effect=replay(env2, 2)
            ),
        ):
            symbolic_shapes._flush_native_queries()
        self.assertEqual(
            order, [(1, sympy.Gt(s1_, 1)), (2, sympy.Gt(s2_, 1)), (1, sympy.Gt(s1_, 0))]
        )

    def test_suppress_guards_mirror(self):
        env, _, _ = self.make_env(allow_native=False)
        self.assertFalse(torch._C._symbolic._suppress_guards())
        with env.suppress_guards():
            self.assertTrue(torch._C._symbolic._suppress_guards())
            with env.suppress_guards():
                self.assertTrue(torch._C._symbolic._suppress_guards())
            self.assertTrue(torch._C._symbolic._suppress_guards())
        self.assertFalse(torch._C._symbolic._suppress_guards())

    @parametrize(
        "name,value",
        [("backed_size_oblivious", True), ("aggressive_guard_free_semantics", 1)],
    )
    def test_live_config(self, name, value):
        env, s, t = self.make_env()
        native = env._native_env
        n = sympy.Gt(s, 1)
        self.assertTrue(torch._C._symbolic._native_config_is_default())
        with symbolic_shapes.config.patch(**{name: value}):
            self.assertFalse(torch._C._symbolic._native_config_is_default())
            self.assertIsNone(native.evaluate_expr(n))
            self.assertEqual(native.static_eval(n)[0], True)
        self.assertTrue(torch._C._symbolic._native_config_is_default())
        self.assertIsNotNone(native.evaluate_expr(n))

    @mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True)
    def test_check_mode(self):
        env, s, t = self.make_env()
        native = env._native_env
        n = sympy.Gt(s, 1)
        native.static_eval(n)
        symbolic_shapes._flush_native_queries()
        # A write that bypasses the notifying container.
        dict.__setitem__(env.var_to_range, s, ValueRanges(3, 9))
        native.static_eval(n)
        with self.assertRaisesRegex(AssertionError, "native mirror"):
            symbolic_shapes._flush_native_queries()
        list.append(env.guards, None)
        native.static_eval(n)
        with self.assertRaisesRegex(AssertionError, "pristine native env"):
            symbolic_shapes._flush_native_queries()

    @parametrize("seed", range(8))
    def test_replay_matches_python(self, seed):
        rng = random.Random(seed)
        steps = [
            (
                rng.random() < 0.5,
                rng.randrange(len(self.QUERY_POOL)),
                rng.random() < 0.3,
                rng.random() < 0.2,
            )
            for _ in range(30)
        ]
        off, want = self.run_steps(False, steps)
        with mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True):
            on, got = self.run_steps(True, steps)
            symbolic_shapes._flush_native_queries()
        self.assertEqual(got, want)
        self.assertEqual(
            [str(g.expr) for g in on.guards], [str(g.expr) for g in off.guards]
        )
        on.check_equal(off)


class TestNativeSymNode(TestCase):
    INT_OPS = [
        "add",
        "sub",
        "mul",
        "floordiv",
        "int_floordiv",
        "mod",
        "sym_min",
        "sym_max",
        "eq",
        "ne",
        "gt",
        "lt",
        "le",
        "ge",
    ]
    GUARDS = [
        "guard_bool",
        "guard_int",
        "bool_",
        "int_",
        "guard_or_false",
        "guard_or_true",
        "statically_known_true",
        "expect_true",
    ]

    def make_env(self, native=True):
        env = ShapeEnv(_allow_native=native)
        syms = [
            env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC),
            env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC),
            env.create_symbol(3, ConstantSource("c"), DimDynamic.DYNAMIC),
            env.create_symbol(
                -4, ConstantSource("d"), DimDynamic.DYNAMIC, positive=None
            ),
        ]
        return env, syms

    @staticmethod
    def pair(env, expr, pytype, hint):
        # A native None hint is no hint; a Python one is computed.
        py_hint = _NO_HINT if hint is None else hint
        native = env._native_env.make_node(expr, pytype, hint)
        return native, SymNode(expr, env, pytype, py_hint)

    @staticmethod
    def node(env, expr, pytype, hint):
        """A native node on a native env, else a Python one."""
        if env._native_env is not None:
            return env._native_env.make_node(expr, pytype, hint)
        return SymNode(expr, env, pytype, _NO_HINT if hint is None else hint)

    def assertSameGuards(self, on_env, off_env):
        self.assertEqual(len(on_env.guards), len(off_env.guards))
        for x, y in zip(on_env.guards, off_env.guards):
            assert_same_function(self, x.expr, y.expr)

    def check(self, n, p):
        assert_same_function(self, n._expr, p._expr)
        self.assertIs(n.pytype, p.pytype)
        self.assertEqual(n.hint, p.hint)
        self.assertIs(type(n.hint), type(p.hint))
        self.assertEqual(n.constant, p.constant)
        # Native nodes never take Python's optimized summation path.
        if isinstance(n, _NativeSymNode):
            self.assertFalse(n._optimized_summation)
        self.assertEqual(n.str(), str(n.expr))
        # Either side may fold to a constant that the other leaves symbolic.
        try:
            want = p.maybe_as_int()
            got = n.maybe_as_int()
        except Exception:
            # int(-int_oo) raises.
            pass
        else:
            if want is not None and got is not None:
                self.assertEqual(got, want, f"{n._expr} vs {p._expr}")

    def call(self, n, p, method, *args):
        """Runs method on both; returns the native result or None if it raised."""
        try:
            want = getattr(p, method)(*(a[1] for a in args))
        except Exception as e:
            with self.assertRaises(type(e)):
                getattr(n, method)(*(a[0] for a in args))
            return None
        got = getattr(n, method)(*(a[0] for a in args))
        self.check(got, want)
        return got, want

    @parametrize("seed", range(8))
    def test_differential(self, seed):
        rng = random.Random(seed)
        env, syms = self.make_env()
        ints = [self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms]
        ints.append(self.pair(env, syms[0], int, None))
        ints += [
            (n.wrap_int(c), p.wrap_int(c)) for c in (0, 1, 2, -3) for n, p in ints[:1]
        ]
        bools = []
        native = 0
        for _ in range(120):
            k = rng.randrange(10)
            if k < 7 or not bools:
                method = rng.choice(self.INT_OPS)
                r = self.call(*rng.choice(ints), method, rng.choice(ints))
            elif k < 8:
                r = self.call(*rng.choice(ints), "neg")
            elif k < 9:
                r = self.call(
                    *rng.choice(bools),
                    rng.choice(["sym_and", "sym_or"]),
                    rng.choice(bools),
                )
            else:
                r = self.call(*rng.choice(bools), "sym_not")
            if r is None or type(r[0]) is SymNode:
                continue
            native += 1
            pool = ints if r[0].pytype is int else bools
            if len(pool) < 16:
                pool.append(r)
            else:
                pool[rng.randrange(len(pool))] = r
        self.assertGreater(native, 80)
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(env._native_env.pristine)

    def test_optimized_summation(self):
        env, syms = self.make_env()
        a, b, c, d = (self.pair(env, s, int, 1) for s in syms)
        ab = self.call(*a, "add", b)
        abc = self.call(*ab, "add", c)
        self.call(*abc, "add", a)
        cd = self.call(*c, "add", d)
        self.call(*ab, "add", cd)
        self.call(*abc, "add", cd)
        self.call(*cd, "add", abc)
        self.call(*a, "add", abc)
        self.call(*ab, "add", ab)

    def test_pow_by_natural(self):
        env, syms = self.make_env()
        a = self.pair(env, syms[0], int, 5)
        for c in (0, 1, 3, -1):
            r = self.call(*a, "pow_by_natural", (a[0].wrap_int(c), a[1].wrap_int(c)))
            self.assertIs(type(r[0]) is SymNode, c < 0)
        big = self.pair(env, syms[1], int, 7)
        r = self.call(
            *big, "pow_by_natural", (big[0].wrap_int(40), big[1].wrap_int(40))
        )
        self.assertIs(type(r[0]), SymNode)

    def test_mod(self):
        env, syms = self.make_env()
        e = env.create_symbol(4, ConstantSource("e"), DimDynamic.DYNAMIC, positive=None)
        a, c, d, e = (self.pair(env, s, int, 4) for s in (syms[0], syms[2], syms[3], e))
        self.assertIs(type(self.call(*a, "mod", c)[0]._expr), Mod)
        self.assertIs(type(self.call(*d, "mod", c)[0]._expr), PythonMod)
        self.assertIs(type(self.call(*c, "mod", d)[0]._expr), PythonMod)
        # a - 2 is nonnegative by range only.
        two = (a[0].wrap_int(2), a[1].wrap_int(2))
        r = self.call(*self.call(*a, "sub", two), "mod", c)
        self.assertIsNot(type(r[0]), SymNode)
        self.assertIs(type(r[0]._expr), Mod)
        # A range answer needs a pristine env.
        env._update_var_to_range(e[1]._expr, ValueRanges(0, 10))
        r = self.call(*e, "mod", c)
        self.assertIs(type(r[0]), SymNode)
        self.assertIs(type(r[0]._expr), Mod)

    @parametrize("mutation", ["range", "replacement", "deepcopy", "pickle"])
    def test_mod_memo(self, mutation):
        # _symop_cache keeps the first Mod/PythonMod choice, which reads ranges.
        import copy
        import pickle

        def mods(cache):
            return {k: v for k, v in cache.items() if k[0] == "mod"}

        caches = []
        for native in (False, True):
            env, syms = self.make_env(native)
            a, c = self.node(env, syms[0], int, 5), self.node(env, syms[2], int, 3)
            # a - 2 is nonnegative by range only.
            x = a.sub(a.wrap_int(2))
            self.assertIs(type(x.mod(c)._expr), Mod)
            if mutation == "pickle":
                caches.append(mods(pickle.loads(pickle.dumps(env._symop_cache))))
                continue
            if mutation == "replacement":
                env._set_replacement(syms[3], sympy.Integer(-4), "test")
                # Native results are Python from here on, and a Python node
                # cannot take a native operand yet.
                c = SymNode(syms[2], env, int, 3)
            else:
                if mutation == "deepcopy":
                    env = copy.deepcopy(env)
                    x = SymNode(syms[0] - 2, env, int, 3)
                    c = SymNode(syms[2], env, int, 3)
                env.var_to_range[syms[0]] = ValueRanges(-10, 10)
            r = x.mod(c)
            self.assertIs(type(r._expr), Mod)
            self.assertIs(type(r) is SymNode, not native or mutation != "range")
            # A new choice reads the current range.
            r = x.sub(x.wrap_int(1)).mod(c)
            self.assertIs(type(r._expr), PythonMod)
            self.assertIs(type(r), SymNode)
            caches.append(mods(env._symop_cache))
        self.assertEqual(caches[0], caches[1])
        want = {"range": 2, "replacement": 3, "deepcopy": 2, "pickle": 1}[mutation]
        self.assertEqual(len(caches[0]), want)

    def test_hint_fallback(self):
        env, syms = self.make_env()
        big = self.pair(env, syms[0], int, 2**62)
        r = self.call(*big, "add", big)
        self.assertIs(type(r[0]), SymNode)
        self.assertEqual(r[0].hint, 2**63)
        zero = (big[0].wrap_int(0), big[1].wrap_int(0))
        self.assertIsNone(self.call(*big, "floordiv", zero))
        self.assertIsNone(self.call(*big, "mod", zero))
        low = self.pair(env, syms[3], int, -(2**63))
        self.assertIs(type(self.call(*low, "neg")[0]), SymNode)
        minus_one = (low[0].wrap_int(-1), low[1].wrap_int(-1))
        self.assertIs(type(self.call(*low, "floordiv", minus_one)[0]), SymNode)
        self.call(*low, "mod", minus_one)

    def test_int_oo(self):
        env, syms = self.make_env()
        a = self.pair(env, syms[0], int, 5)
        oo = self.pair(env, int_oo, int, None)
        for method in ("add", "sub", "mul", "sym_max", "lt"):
            self.call(*a, method, oo)
            self.call(*oo, method, a)
        self.call(*oo, "neg")
        self.assertEqual(oo[0].str(), oo[1].str())

    def test_unbacked(self):
        env, _ = self.make_env()
        u = env.create_unbacked_symint().node.expr
        n, p = self.pair(env, u, int, None)
        three = (n.wrap_int(3), p.wrap_int(3))
        for method in ("eq", "ne", "lt", "ge"):
            self.call(n, p, method, three)
            self.call(*three, method, (n, p))
        self.assertIsNone(self.call(n, p, "add", three)[0].hint)

    def test_python_fallbacks(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, 5) for s in syms[:2])
        for method in ("truediv", "float_truediv", "int_truediv", "pow", "float_pow"):
            r = self.call(*a, method, b)
            self.assertIs(type(r[0]), SymNode)
        for method in ("ceil", "floor", "sym_float"):
            self.assertIs(type(self.call(*a, method)[0]), SymNode)
        f = a[0].wrap_float(2.5)
        self.assertIs(type(f), SymNode)
        self.assertEqual(f.constant, 2.5)
        # A Python operand makes the op Python.
        r = self.call(a[0], a[1], "add", (b[1], b[1]))
        self.assertIs(type(r[0]), SymNode)
        # So does an int operand of a logical op.
        t = self.call(*a, "lt", b)
        self.assertIs(type(self.call(*t, "sym_and", a)[0]), SymNode)

    SIZES_STRIDES = [
        "is_contiguous",
        "is_channels_last_contiguous_2d",
        "is_channels_last_contiguous_3d",
        "is_channels_last_strides_2d",
        "is_channels_last_strides_3d",
        "is_non_overlapping_and_dense_indicator",
        "is_non_overlapping_and_dense",
    ]

    def int_pairs(self, env, syms):
        a, b, c, d = syms
        exprs = [a, b, c, d, 2 * a, a + 1, a * b, b * c, a * b * c]
        vals = env.backed_var_to_val
        pairs = [self.pair(env, e, int, int(e.xreplace(vals))) for e in exprs]
        n, p = pairs[0]
        return pairs + [(n.wrap_int(k), p.wrap_int(k)) for k in (0, 1, 2, 3)]

    @parametrize("seed", range(8))
    def test_sizes_strides(self, seed):
        rng = random.Random(seed)
        env, syms = self.make_env()
        pool = self.int_pairs(env, syms)
        native = 0
        for _ in range(60):
            dim = rng.choice([1, 2, 3, 4, 4, 5, 5])
            if rng.random() < 0.3:
                # Contiguous strides make the relationals fold.
                sizes = [rng.choice(pool) for _ in range(dim)]
                strides = [pool[-3]]
                for s in reversed(sizes[1:]):
                    strides.insert(0, self.call(*strides[0], "mul", s))
            else:
                sizes = [rng.choice(pool) for _ in range(dim)]
                strides = [rng.choice(pool) for _ in range(dim)]
            base = rng.choice(sizes + strides)
            method = rng.choice(self.SIZES_STRIDES)
            r = self.call(
                *base,
                method,
                ([x[0] for x in sizes], [x[1] for x in sizes]),
                ([x[0] for x in strides], [x[1] for x in strides]),
            )
            if r is not None and type(r[0]) is _NativeSymNode:
                native += 1
        for method in self.SIZES_STRIDES:
            self.call(*pool[0], method, ([], []), ([], []))
        self.assertGreater(native, 50)
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(env._native_env.pristine)

    def test_sizes_strides_fallback(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms[:2])
        one = (a[0].wrap_int(1), a[1].wrap_int(1))
        no_hint = self.pair(env, syms[0], int, None)
        py_b = (b[1], b[1])
        for sizes, strides in [
            ([a, py_b], [b, one]),  # Python operand
            ([a, no_hint], [b, one]),  # Python computes a hint
            ([a, b], [b]),  # unequal lengths
        ]:
            sizes = tuple(list(x) for x in zip(*sizes))
            strides = tuple(list(x) for x in zip(*strides))
            for method in self.SIZES_STRIDES:
                r = self.call(*a, method, sizes, strides)
                if r is not None:
                    self.assertIs(type(r[0]), SymNode)

    def test_sizes_strides_tensor(self):
        # SymbolicShapeMeta calls the virtuals on the first symbolic node.
        def run(native):
            env, syms = self.make_env(native)
            a, b = (
                torch.SymInt(self.node(env, s, int, int(env.backed_var_to_val[s])))
                for s in syms[:2]
            )
            t = torch.empty_strided((a, b, 3), (3 * b, 3, 1), device="meta")
            u = torch.empty_strided(
                (2, a, b, 3), (3 * a * b, 1, 3 * a, a), device="meta"
            )
            out = []
            for x in (t, t.transpose(0, 2), u, u.transpose(1, 3)):
                out += [
                    x.is_contiguous(),
                    x.is_contiguous(memory_format=torch.channels_last),
                    # preserve_format asks is_non_overlapping_and_dense.
                    str(torch.empty_like(x).stride()),
                ]
            return out, [str(g.expr) for g in env.guards]

        self.assertEqual(run(True), run(False))

    def test_sym_sum(self):
        env, syms = self.make_env()
        pool = self.int_pairs(env, syms)
        no_hint = self.pair(env, syms[1], int, None)
        for args in [
            pool[:3],
            pool[4:9],
            [pool[0], pool[-1], pool[0]],
            [no_hint, pool[0]],
        ]:
            r = self.call(
                *args[0], "sym_sum", ([x[0] for x in args], [x[1] for x in args])
            )
            self.assertIs(type(r[0]), _NativeSymNode)
        # A Python operand makes it Python.
        r = self.call(
            *pool[0], "sym_sum", ([pool[0][0], pool[1][1]], [pool[0][1], pool[1][1]])
        )
        self.assertIs(type(r[0]), SymNode)
        # int64 overflow of the hint falls back to Python's bigint.
        big = (pool[0][0].wrap_int(2**62), pool[0][1].wrap_int(2**62))
        args = [big, big, pool[0]]
        r = self.call(*pool[0], "sym_sum", ([x[0] for x in args], [x[1] for x in args]))
        self.assertIs(type(r[0]), SymNode)
        x, y = (torch.SymInt(n) for n, _ in pool[:2])
        self.assertIs(type(torch.sym_sum([x, y, 3]).node), _NativeSymNode)
        self.assertEqual(len(env.guards), 0)

    def test_wrap(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        self.check(n.wrap_int(3), p.wrap_int(3))
        self.check(n.wrap_bool(True), p.wrap_bool(True))
        self.assertIsNone(n.maybe_as_int())
        self.assertEqual(n.wrap_int(-2).maybe_as_int(), -2)
        self.assertTrue(n.has_hint())
        self.assertTrue(n.is_int())
        self.assertFalse(n.is_bool())
        self.assertEqual(str(n), str(syms[0]))

    def test_replacements(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, 5) for s in syms[:2])
        env._set_replacement(syms[1], syms[0] + 2, "test")
        r = self.call(*a, "add", b)
        self.assertIs(type(r[0]), SymNode)
        self.assertEqual(b[0].str(), b[1].str())
        self.assertIs(type(self.call(*b, "neg")[0]), SymNode)

    def test_guards(self):
        env, syms = self.make_env()
        native = env._native_env
        a, b = (native.make_node(s, int, h) for s, h in zip(syms, (5, 7)))
        two = a.wrap_int(2)
        # s0 >= 2 by range.
        ge = a.ge(two)
        self.assertTrue(ge.guard_bool("", 0))
        self.assertTrue(ge.bool_())
        self.assertTrue(ge.guard_or_false("", 0))
        self.assertTrue(ge.guard_or_true("", 0))
        self.assertTrue(ge.statically_known_true("", 0))
        self.assertTrue(ge.expect_true("", 0))
        # s0 + s1 >= 4 by static evaluation.
        self.assertTrue(a.add(b).ge(a.wrap_int(4)).guard_bool("", 0))
        self.assertEqual(a.wrap_int(3).int_(), 3)
        self.assertFalse(a.wrap_int(0).bool_())
        self.assertTrue(a.wrap_bool(True).guard_bool("", 0))
        # The hint disproves it without a query.
        self.assertFalse(a.lt(two).statically_known_true("", 0))
        eq = a.eq(a.wrap_int(5))
        self.assertFalse(eq.statically_known_true("", 0))
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(native.pristine)
        s0 = syms[0]
        rt = native.arena.roundtrip
        ge_e, eq_e = rt(sympy.Ge(s0, 2)), rt(sympy.Eq(s0, 5))
        self.assertEqual(
            [q[1:] for q in native.take_queries()],
            [
                (ge_e, True, True, False, False, sympy.true),
                (ge_e, True, True, True, False, sympy.true),
                (ge_e, False, None, None, False, sympy.true),
                (ge_e, True, True, None, False, sympy.true),
                (rt(sympy.Ge(s0 + syms[1], 4)), True, True, None, False, sympy.true),
                (sympy.Integer(3), True, 3, None, False, sympy.Integer(3)),
                (sympy.Integer(0), True, 0, None, False, sympy.Integer(0)),
                (sympy.true, True, True, None, False, sympy.true),
                (eq_e, False, None, None, False, None),
            ],
        )

        # Unknown statically: Python guards, and native answers stop.
        self.assertTrue(eq.guard_or_false("", 0))
        self.assertEqual([g.expr for g in env.guards], [eq_e])
        self.assertTrue(ge.guard_bool("", 0))
        self.assertEqual(native.take_queries(), [])
        with self.assertRaisesRegex(AssertionError, "bool"):
            a.guard_or_false("", 0)
        with self.assertRaisesRegex(AssertionError, "bool"):
            a.statically_known_true("", 0)
        self.assertEqual(a.int_(), 5)

    def test_guard_no_hint(self):
        env, syms = self.make_env()
        n, p = self.pair(env, sympy.Ge(syms[0], 2), bool, None)
        # expect_true without a hint defers to Python.
        self.assertTrue(n.expect_true("", 0))
        self.assertEqual(env._native_env.take_queries(), [])
        self.assertTrue(n.guard_bool("", 0))
        self.assertEqual(len(env._native_env.take_queries()), 1)

    @parametrize(
        "name,value",
        [
            ("backed_size_oblivious", True),
            ("aggressive_guard_free_semantics", 1),
            ("aggressive_guard_free_semantics", 2),
        ],
    )
    def test_guard_live_config(self, name, value):
        results = []
        for native in (True, False):
            env, syms = self.make_env(native)
            ge, eq = (sympy.Ge(syms[0], 2), sympy.Eq(syms[0], 5))
            nodes = [
                self.node(env, ge, bool, True),
                self.node(env, eq, bool, True),
                # Level 2 returns the fallback value without range analysis.
                self.node(env, ge, bool, None),
            ]
            with symbolic_shapes.config.patch(**{name: value}):
                results.append(
                    (
                        [n.guard_or_false("", 0) for n in nodes],
                        [n.guard_or_true("", 0) for n in nodes],
                        [str(g.expr) for g in env.guards],
                    )
                )
            if native:
                self.assertEqual(env._native_env.take_queries(), [])
        self.assertEqual(results[0], results[1])

    def guard_program(self, native, seed):
        """Short random programs of ops then guards, each on a fresh env."""
        rng = random.Random(seed)
        answers, envs = [], []
        for _ in range(12):
            env, syms = self.make_env(native)
            envs.append(env)
            ints = [self.node(env, s, int, int(env.backed_var_to_val[s])) for s in syms]
            ints += [ints[0].wrap_int(c) for c in (0, 1, 2, 5, -3)]
            bools = [ints[0].wrap_bool(True)]
            for _ in range(rng.randrange(1, 8)):
                a, b = rng.choice(ints), rng.choice(ints)
                if type(a) is SymNode and type(b) is not SymNode:
                    # A Python SymNode op does not take native operands yet.
                    b = SymNode(b._expr, env, int, b.hint, constant=b.constant)
                try:
                    r = getattr(a, rng.choice(self.INT_OPS))(b)
                except Exception as e:
                    answers.append(type(e))
                    continue
                (ints if r.pytype is int else bools).append(r)
            for _ in range(rng.randrange(1, 6)):
                method = rng.choice(self.GUARDS)
                n = rng.choice(bools if rng.random() < 0.8 else ints)
                args = () if method in ("bool_", "int_") else ("", 0)
                suppress = rng.random() < 0.2
                with env.suppress_guards() if suppress else contextlib.nullcontext():
                    try:
                        answers.append((method, getattr(n, method)(*args)))
                    except Exception as e:
                        answers.append((method, type(e)))
        return answers, envs

    @parametrize("seed", range(8))
    def test_guard_differential(self, seed):
        python_calls = []

        def counted(fn):
            def wrapper(*args, **kwargs):
                python_calls.append(fn.__name__)
                return fn(*args, **kwargs)

            return wrapper

        with (
            mock.patch.object(
                ShapeEnv, "evaluate_sym_node", counted(ShapeEnv.evaluate_sym_node)
            ),
            mock.patch.object(
                symbolic_shapes,
                "_static_eval_sym_bool",
                counted(symbolic_shapes._static_eval_sym_bool),
            ),
        ):
            want, off = self.guard_program(False, seed)
            off_calls = len(python_calls)
            python_calls.clear()
            with mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True):
                got, on = self.guard_program(True, seed)
                symbolic_shapes._flush_native_queries()
        self.assertEqual(got, want)
        # Native expressions are not structurally sympy's, so guards are
        # compared by value.
        for on_env, off_env in zip(on, off):
            self.assertEqual(len(on_env.guards), len(off_env.guards))
            for g_on, g_off in zip(on_env.guards, off_env.guards):
                assert_same_function(self, g_on.expr, g_off.expr)
        self.assertGreater(off_calls - len(python_calls), 10)

    def trace(self, native, fn, pre_dispatch):
        env, syms = self.make_env(native)
        a, b = (
            torch.SymInt(self.node(env, s, int, int(env.backed_var_to_val[s])))
            for s in syms[:2]
        )
        gm = make_fx(fn, tracing_mode="real", pre_dispatch=pre_dispatch)(a, b)
        return env, gm

    @parametrize("pre_dispatch", [False, True])
    def test_proxy_make_fx(self, pre_dispatch):
        def f(a, b):
            c = a + b
            d = c * 2 - a // b
            e = (d % 3 + 1) * 1
            m = torch.sym_max(a, b) + torch.sym_min(a, 3)
            one = a.node.wrap_int(1)
            sizes, strides = [a.node, b.node], [b.node, one]
            contig = torch.SymBool(a.node.is_contiguous(sizes, strides))
            dense = torch.SymBool(a.node.is_non_overlapping_and_dense(sizes, strides))
            ite = torch.sym_ite(a < b, a, b)
            total = torch.sym_sum([a, b, 3])
            return (
                e,
                m,
                a**2,
                a / b,
                torch.sym_float(a),
                -a,
                ite,
                contig,
                dense,
                total,
            )

        def g(a, b):
            lt = a < b
            return (
                lt & (b > 2),
                lt | (a == b),
                torch.sym_not(lt),
                a != b,
                a <= b,
                a >= b,
            )

        for fn in (f, g):
            on_env, on = self.trace(True, fn, pre_dispatch)
            off_env, off = self.trace(False, fn, pre_dispatch)
            self.assertEqual(on.code, off.code)
            self.assertSameGuards(on_env, off_env)
            on_vals = [n.meta.get("val") for n in on.graph.nodes]
            off_vals = [n.meta.get("val") for n in off.graph.nodes]
            for x, y in zip(on_vals, off_vals, strict=True):
                if isinstance(y, torch.SymInt | torch.SymBool | torch.SymFloat):
                    assert_same_function(self, x.node.expr, y.node.expr)
            ops = [n for n in on.graph.nodes if n.op == "call_function"]
            self.assertIsInstance(ops[0].meta["val"].node, _NativeSymNode)

        # The inner op of the dispatch runs natively.
        _, gm = self.trace(True, lambda a, b: a * b + 1, pre_dispatch)
        vals = [n.meta["val"] for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(all(isinstance(v.node, _NativeSymNode) for v in vals))

    def test_proxy_pre_dispatch_excluded(self):
        # get_proxy_mode() reads the global pre-dispatch slot even with
        # PreDispatch removed from the include set (torch._export.wrappers).
        def f(a, b):
            PreDispatch = torch._C.DispatchKey.PreDispatch
            include = torch._C._dispatch_tls_local_include_set().remove(PreDispatch)
            exclude = (
                torch._C._dispatch_tls_local_exclude_set()
                | torch._C.DispatchKeySet(PreDispatch)
            )
            with torch._C._ForceDispatchKeyGuard(include, exclude):
                return a * b + 1

        _, on = self.trace(True, f, True)
        _, off = self.trace(False, f, True)
        self.assertIn("mul", off.code)
        self.assertEqual(on.code, off.code)

    def test_expr(self):
        env, syms = self.make_env()
        n = env._native_env.make_node(syms[0], int, 5)
        self.assertEqual(n.add(n).expr, 2 * syms[0])

    def test_proxy_mul_by_one(self):
        # __sym_dispatch__ returns the operand itself.
        _, gm = self.trace(True, lambda a, b: (a * 1, 1 * b), False)
        self.assertNotIn("mul", gm.code)

    def test_proxy_sym_dispatch(self):
        # ProxyTorchDispatchMode.__sym_dispatch__ runs natively, including a
        # constant lhs, which Python evaluates through the reflected or
        # mirrored method of the rhs.
        def f(a, b):
            lt = a < b
            w = a.node.wrap_int
            return (
                3 + a,
                3 - a,
                2 * a,
                7 // a,
                7 % a,
                a * b - b,
                3 < a,
                3 > a,
                3 <= a,
                3 >= a,
                3 == a,
                3 != a,
                a == 3,
                torch.sym_max(3, a),
                torch.sym_min(3, a),
                torch.sym_max(a, 3),
                torch.sym_max(a, a),
                True & lt,
                False | lt,
                lt & True,
                lt | (a == b),
                -a,
                torch.sym_not(lt),
                w(3).add(w(4)).constant,
                w(3).lt(w(4)).constant,
            )

        sym_dispatch = mock.patch.object(
            proxy_tensor,
            "handle_sym_dispatch",
            wraps=proxy_tensor.handle_sym_dispatch,
        )
        sym_register = mock.patch.object(
            proxy_tensor, "_sym_register", wraps=proxy_tensor._sym_register
        )
        with sym_dispatch as m, sym_register as r:
            on_env, on = self.trace(True, f, False)
        # Only the two-constant ops, which are Python arithmetic.
        self.assertEqual(m.call_count, 2)
        # Those two, and sym_max(a, a), whose Symbol result set_proxy_slot tracks.
        self.assertEqual(r.call_count, 3)
        off_env, off = self.trace(False, f, False)
        self.assertEqual(on.code, off.code)
        self.assertSameGuards(on_env, off_env)
        for x, y in zip(on.graph.nodes, off.graph.nodes, strict=True):
            self.assertEqual(list(x.meta), list(y.meta))
            vx, vy = x.meta.get("val"), y.meta.get("val")
            if isinstance(vy, torch.SymInt | torch.SymBool | torch.SymFloat):
                assert_same_function(self, vx.node.expr, vy.node.expr)
            else:
                self.assertEqual(str(vx), str(vy))

        # MAGIC logging keeps the Python dispatch.
        def logged(native):
            with self.assertLogs(sym_node.sym_node_log, level="DEBUG") as logs:
                _, gm = self.trace(native, lambda a, b: a * b + 1, False)
            return gm.code, [x for x in logs.output if "MAGIC" in x]

        with sym_dispatch as m:
            on = logged(True)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(on, logged(False))

    def test_proxy_torch_function(self):
        # sym_min/sym_max/sym_not dispatch through torch functions, which a
        # TorchFunctionMode sees.
        class Spy(torch.overrides.TorchFunctionMode):
            def __init__(self):
                super().__init__()
                self.funcs = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.funcs.append(func.__name__)
                return func(*args, **(kwargs or {}))

        def traced(native):
            spy = Spy()

            def f(a, b):
                with spy:
                    return a.__sym_max__(b), a.__sym_min__(b), (a < b).__sym_not__()

            _, gm = self.trace(native, f, False)
            return gm.code, spy.funcs

        on_code, on_funcs = traced(True)
        self.assertEqual(on_code, traced(False)[0])
        # Not compared to off: the Python SymNode also calls torch.sym_max and
        # torch.sym_min on the hints.
        self.assertEqual(on_funcs, ["sym_max", "sym_min", "sym_not"])

    def test_proxy_hint_raises(self):
        # binary_magic_impl computes the hint before dispatching.
        env, syms = self.make_env()
        a = torch.SymInt(env._native_env.make_node(syms[0], int, 5))
        zero = torch.SymInt(env._native_env.make_node(syms[0] - 5, int, 0))
        with self.assertRaises(ZeroDivisionError):
            make_fx(lambda x, y: x // y, tracing_mode="real")(a, zero)

    def test_proxy_direct(self):
        # Native operands skip the Python impl unless its hint computation may
        # raise, or capture_provenance logs.
        sym_node = torch.fx.experimental.sym_node
        spy = mock.patch.object(
            sym_node, "method_to_operator", wraps=sym_node.method_to_operator
        )
        fn = lambda a, b: (a * b + 1, -a, a < b, torch.sym_not(a < b), a % b, a**b)  # noqa: E731
        with spy as m:
            _, gm = self.trace(True, fn, False)
        self.assertEqual([c.args for c in m.call_args_list], [("pow_by_natural",)])
        self.assertEqual(gm.code, self.trace(False, fn, False)[1].code)

        env, syms = self.make_env()
        a = torch.SymInt(env._native_env.make_node(syms[0], int, 5))
        zero = torch.SymInt(env._native_env.make_node(syms[0] - 5, int, 0))
        with (
            spy as m,
            mock.patch.object(torch._logging._internal, "GET_DTRACE_STRUCTURED", True),
            mock.patch.object(sym_node, "dtrace_structured"),
        ):
            make_fx(lambda x: -(x + x), tracing_mode="real")(a)
        self.assertEqual([c.args for c in m.call_args_list], [("add",), ("neg",)])
        with self.assertRaises(ZeroDivisionError):
            make_fx(lambda x, y: x % y, tracing_mode="real")(a, zero)

    def test_expr_facts(self):
        # has_free_symbols and fetch_sym_proxy read expr.is_number / is_Boolean.
        env, syms = self.make_env()
        a, b, c = (self.node(env, s, int, h) for s, h in zip(syms, (5, 7, 3)))
        lt = a.lt(b)
        nodes = [
            a,
            a.add(b),
            a.sub(a),
            a.mul(a.wrap_int(3)),
            a.int_floordiv(b),
            a.wrap_int(4).int_floordiv(a.wrap_int(3)),
            a.sym_max(b),
            lt,
            a.eq(a),
            a.eq(a.add(a.wrap_int(1))),
            lt.sym_and(b.lt(c)),
            lt.sym_or(b.lt(c)),
            lt.sym_and(b.lt(c)).sym_not(),
            self.node(env, sympy.Integer(3), int, 3),
            self.node(env, sympy.true, bool, True),
            b.mul(b),
        ]
        for n in nodes:
            self.assertIsInstance(n, _NativeSymNode)
            self.assertIs(n._expr_is_number, n.expr.is_number, n.expr)
            self.assertIs(n._expr_is_Boolean, n.expr.is_Boolean, n.expr)
        self.assertEqual(
            [symbolic_shapes.has_free_symbols(torch.SymInt(n)) for n in nodes[:7]],
            [True, True, False, True, True, False, True],
        )
        env._set_replacement(syms[1], sympy.Integer(7), "test")
        for n in nodes:
            self.assertIs(n._expr_is_number, n.expr.is_number, n.expr)
            self.assertIs(n._expr_is_Boolean, n.expr.is_Boolean, n.expr)
        self.assertFalse(symbolic_shapes.has_free_symbols(torch.SymInt(nodes[-1])))

        # A numeric SymInt enters the graph as its value.
        fn = lambda a, b: torch.ones(2) + (a - a)  # noqa: E731
        gm = self.trace(True, fn, False)[1]
        self.assertEqual(gm.code, self.trace(False, fn, False)[1].code)
        self.assertIn("add = torch.ops.aten.add.Tensor(ones, 0)", gm.code)

    def test_make_node(self):
        env, syms = self.make_env()
        native = env._native_env
        with self.assertRaisesRegex(RuntimeError, "int or bool"):
            native.make_node(syms[0], float, 5.0)
        with self.assertRaisesRegex(RuntimeError, "pytype"):
            native.make_node(syms[0], int, True)
        with self.assertRaisesRegex(RuntimeError, "pytype"):
            native.make_node(sympy.Eq(syms[0], 1), bool, 1)
        with self.assertRaises(NativeUnsupported):
            native.make_node(sympy.Float(1.5), int, None)

    def test_lifetime(self):
        # A native node holds its ShapeEnv, as a Python SymNode does.
        import gc
        import weakref

        env, syms = self.make_env()
        ref = weakref.ref(env)
        n = env._native_env.make_node(syms[0], int, 5)
        del n
        m = env._native_env.make_node(syms[1], int, 7)
        n = m.add(m)
        del env
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIs(n.shape_env, ref())
        self.assertEqual(n.truediv(m).hint, 2.0)
        del m
        gc.collect()
        self.assertIsNotNone(ref())
        del n
        gc.collect()
        self.assertIsNone(ref())
        env, syms = self.make_env()
        native = env._native_env
        del env
        gc.collect()
        n = native.make_node(syms[0], int, 5)
        self.assertIsNone(n.shape_env)
        with self.assertRaisesRegex(
            RuntimeError, "ShapeEnv of a native SymNode is gone"
        ):
            n.truediv(n)

    def test_create_symintnode(self):
        env, syms = self.make_env()
        s = syms[0]
        x = env.create_symintnode(s, hint=5)
        n = x.node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertEqual((n._expr, n.hint), (s, 5))
        want = SymNode(s, env, int, 5)
        self.assertEqual((str(n), repr(n)), (str(want), repr(want)))
        # A backed node without a hint gets the computed one.
        n = env.create_symintnode(s * syms[1], hint=None).node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertEqual(n.hint, 35)
        n = env.create_symboolnode(sympy.Eq(s, 5)).node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertIs(n.hint, True)
        self.assertEqual(env.create_symintnode(sympy.Integer(3), hint=3), 3)
        big = env.create_symbol(2**70, ConstantSource("big"), DimDynamic.DYNAMIC)
        u = env.create_unbacked_symint().node.expr
        m2, mx = sympy.Mul(s, 2, evaluate=False), sympy.Max(s, syms[1])
        for sym, hint in ((big, 2**70), (u, None), (s, x), (5, 5), (m2, 10), (mx, 7)):
            self.assertIs(type(env.create_symintnode(sym, hint=hint).node), SymNode)
        self.assertIs(type(env.create_symintnode(5, hint=5).node.expr), int)
        flag_off, syms = self.make_env(native=False)
        n = flag_off.create_symintnode(syms[0], hint=5).node
        self.assertIs(type(n), SymNode)

    def test_attributes(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        self.assertIs(n.shape_env, env)
        self.assertIs(n.fx_node, p.fx_node)
        self.assertEqual(n._hint, p._hint)
        self.assertEqual(repr(n), repr(p))
        self.assertIsNone(self.pair(env, syms[3], int, None)[0]._hint)
        self.assertTrue(n._value_eq(p))
        self.assertEqual(n._value_hash(), p._value_hash())
        w = n.with_shape_env(env)
        self.assertIs(type(w), SymNode)
        self.check(w, p)
        with self.assertRaises(AttributeError):
            n.nonexistent
        with self.assertRaises(AttributeError):
            n.__add__

    def test_python_methods(self):
        # Methods without a native binding run the Python impl on the native node.
        env, syms = self.make_env()
        a, d = self.pair(env, syms[0], int, 5), self.pair(env, syms[3], int, -4)
        t = self.call(*a, "lt", d)
        binary = ("lshift", "rshift", "bitwise_and", "bitwise_or", "bitwise_xor")
        for x in (a, d):
            for method in ("abs", "pos", "trunc", "round", "is_integer", "sym_sqrt"):
                self.call(*x, method)
            for method in binary:
                self.call(*x, method, a)
            self.assertIsNone(x[0].maybe_as_float())
        self.assertIsNone(t[0].maybe_as_bool())
        self.assertIs(t[0].wrap_bool(True).maybe_as_bool(), True)
        self.call(*t, "xor", t)
        self.assertEqual(len(env.guards), 0)
        self.assertEqual(a[0].evaluate(), a[1].evaluate())
        self.assertEqual(len(env.guards), 1)

    def test_python_lhs(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms[:2])
        for method in self.INT_OPS + ["truediv", "pow"]:
            self.check(getattr(a[1], method)(b[0]), getattr(a[1], method)(b[1]))
        t = self.call(*a, "lt", b)
        self.check(t[1].sym_ite(a[0], b[0]), t[1].sym_ite(a[1], b[1]))
        ab = torch.SymInt(a[1]) * torch.SymInt(b[0])
        self.assertEqual(str(ab), str(syms[0] * syms[1]))

        # C++ PythonSymNodeImpl ops with a native operand.
        def run(native):
            env, syms = self.make_env(native)
            x = torch.SymInt(SymNode(syms[0], env, int, 5))
            y = torch.SymInt(self.node(env, syms[1], int, 7))
            t = torch.empty((x, y), device="meta")
            return str(t.numel()), str(t.stride()), str(torch.sym_ite(x < y, x, y))

        self.assertEqual(run(True), run(False))

    def test_casters(self):
        env, syms = self.make_env()
        n = env._native_env.make_node(syms[0], int, 5)
        t = n.lt(n.wrap_int(9))
        self.assertIs(type(t), _NativeSymNode)
        b = torch._C._symbolic._roundtrip_symbool(torch.SymBool(t))
        self.assertIs(b.node, t)
        f = torch._C._symbolic._roundtrip_symfloat(torch.SymFloat(n.sym_float()))
        self.assertEqual(str(f), f"ToFloat({syms[0]})")
        x = torch.SymInt(n)
        self.assertIs(x.node, x.node)
        t = torch.empty((x, 2), device="meta")
        self.assertIs(t.size(0).node, n)
        self.assertIs(t.size(0).node, t.size(0).node)

    def test_copy(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        memo = {id(env): env}
        c = copy.deepcopy(n, memo)
        self.assertIs(type(c), SymNode)
        self.assertIs(c.shape_env, env)
        self.check(c, copy.deepcopy(p, memo))
        self.assertIs(type(copy.deepcopy(n).shape_env), ShapeEnv)
        self.assertIs(copy.copy(n).shape_env, env)
        self.check(pickle.loads(pickle.dumps(n)), pickle.loads(pickle.dumps(p)))

    def test_module_helpers(self):
        from torch.fx.experimental.symbolic_shapes import (
            _iterate_nodes,
            free_symbols,
            guard_or_false,
            guard_or_true,
            guarding_hint_or_throw,
            has_free_symbols,
            statically_known_false,
            statically_known_true,
        )

        def run(native):
            env, syms = self.make_env(native)
            a, b = (self.node(env, s, int, h) for s, h in zip(syms, (5, 7)))
            no_hint = self.node(env, syms[0] + syms[1], int, None)
            out = [
                guarding_hint_or_throw(a),
                guarding_hint_or_throw(no_hint),
                list(_iterate_nodes([a, torch.SymInt(b)])) == [a, b],
                str(free_symbols([torch.SymInt(a), b])),
                has_free_symbols(a),
            ]
            # Native nodes do not cache a computed hint.
            self.assertEqual(no_hint._hint, None if native else 12)
            bools = [
                a.ge(a.wrap_int(2)),
                a.lt(a.wrap_int(2)),
                a.add(b).ge(a.wrap_int(4)),
                a.eq(a.wrap_int(5)),
                a.lt(b),
                a.gt(b),
            ]
            out += [statically_known_true(torch.SymBool(t)) for t in bools]
            # Answered natively: queries wait in the replay log.
            pending = torch._C._symbolic._native_queries_pending()
            self.assertEqual(pending, native)
            out += [statically_known_false(torch.SymBool(t)) for t in bools]
            out.append([g.expr for g in env.guards])
            out.append(guard_or_false(torch.SymBool(bools[0])))
            pending = torch._C._symbolic._native_queries_pending()
            self.assertEqual(pending, native)
            for fn in (guard_or_false, guard_or_true, statically_known_true):
                out += [fn(torch.SymBool(t)) for t in bools]
            out.append([g.expr for g in env.guards])
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                out += [guard_or_true(torch.SymBool(t)) for t in bools]
            return out

        self.assertEqual(run(True), run(False))

    def test_no_plain_symnode_isinstance(self):
        # A native node is not a Python SymNode: type checks must use SymNodeTypes.
        import ast

        # Native nodes are never unbacked, so these checks are False for them.
        allowed = {
            ("torch/fx/experimental/symbolic_shapes.py", "_advise_is_size"),
            ("torch/fx/experimental/symbolic_shapes.py", "_advise_is_bounded"),
        }
        root = os.path.dirname(os.path.dirname(torch.__file__))
        found = set()

        class Visitor(ast.NodeVisitor):
            def __init__(self, path):
                self.path = path
                self.func = None

            def visit_FunctionDef(self, node):
                outer, self.func = self.func, node.name
                self.generic_visit(node)
                self.func = outer

            def visit_Call(self, node):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in ("isinstance", "issubclass")
                    and len(node.args) == 2
                ):
                    for n in ast.walk(node.args[1]):
                        name = getattr(n, "id", None) or getattr(n, "attr", None)
                        if name == "SymNode":
                            found.add((self.path, self.func))
                self.generic_visit(node)

        for dirpath, _, files in os.walk(os.path.join(root, "torch")):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                if "SymNode" in src:
                    Visitor(os.path.relpath(path, root)).visit(ast.parse(src))
        self.assertEqual(found, allowed)


class TestNativeSymNodeCompile(TestCase):
    """torch.compile with dynamic shapes, flag on vs off."""

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.fc1(x))
            return torch.softmax(self.fc2(h), dim=-1).view(x.shape[0], -1)

    @staticmethod
    def attention(q, k):
        b, t, d = q.shape
        s = (q @ k.transpose(1, 2)) / d**0.5
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        return (s @ k).reshape(b, t * d)[:, : t * d // 2]

    @staticmethod
    def size_branch(x, y):
        n = x.shape[0]
        z = torch.cat([x, y[: n // 2]])
        if n % 2 == 0 and z.shape[0] > 6:
            return z.flatten()[n:] * n
        return z.sum(0) - n

    CASES = {
        "mlp": (Mlp, lambda n: (torch.randn(n, 3, 8),), (4, 6, 9)),
        "attention": (
            lambda: TestNativeSymNodeCompile.attention,
            lambda n: (torch.randn(2, n, 4), torch.randn(2, n, 4)),
            (5, 8),
        ),
        "size_branch": (
            lambda: TestNativeSymNodeCompile.size_branch,
            lambda n: (torch.randn(n, 3), torch.randn(n + 2, 3)),
            (4, 8, 7, 10),
        ),
    }

    def run_compiled(self, case, native):
        from torch._dynamo.backends.common import aot_autograd
        from torch._guards import detect_fake_mode

        make_fn, make_args, sizes = self.CASES[case]
        torch.manual_seed(0)
        fn = make_fn()
        graphs, envs, vals = [], [], []

        def fw_compiler(gm, example_inputs):
            graphs.append(gm.code)
            return gm.forward

        def backend(gm, example_inputs):
            graphs.append(gm.code)
            envs.append(detect_fake_mode(example_inputs).shape_env)
            vals.extend(
                n.meta["example_value"]
                for n in gm.graph.nodes
                if isinstance(n.meta.get("example_value"), torch.SymInt)
            )
            return aot_autograd(fw_compiler=fw_compiler)(gm, example_inputs)

        torch._dynamo.reset()
        with torch._dynamo.config.patch(use_cpp_symnode=native):
            compiled = torch.compile(fn, backend=backend, dynamic=True)
            outs = []
            for n in sizes:
                args = make_args(n)
                outs.append((compiled(*args), fn(*args)))
        state = [
            (
                [str(g.expr) for g in env.guards],
                sorted(map(str, env.replacements.items())),
                sorted(map(str, env.var_to_range.items())),
            )
            for env in envs
        ]
        return graphs, state, outs, vals, envs

    @parametrize("case", list(CASES))
    def test_compile(self, case):
        graphs, state, outs, vals, envs = self.run_compiled(case, native=True)
        want_graphs, want_state, _, want_vals, _ = self.run_compiled(case, native=False)
        self.assertEqual(graphs, want_graphs)
        self.assertEqual(state, want_state)
        for got, want in outs:
            self.assertEqual(got, want)
        self.assertTrue(all(env._native_env is not None for env in envs))
        self.assertTrue(vals)
        self.assertTrue(all(isinstance(v.node, _NativeSymNode) for v in vals))
        self.assertEqual(
            [(str(v), v.node.hint) for v in vals],
            [(str(v), v.node.hint) for v in want_vals],
        )
        self.assertTrue(not any(isinstance(v.node, _NativeSymNode) for v in want_vals))


@contextlib.contextmanager
def python_glue():
    """Puts the Python magic methods back for the duration."""
    for (cls, attr), fn in sym_node._native_glue_originals.items():
        setattr(cls, attr, fn)
    try:
        yield
    finally:
        sym_node._install_native_glue()


class TestNativeSymIntGlue(TestCase):
    ENTRIES = [
        (cls, attr, kind)
        for cls, attr, _, kind in sym_node._user_magic_entries
        if cls is not torch.SymFloat and kind in ("unary", "binary", "rbinary")
    ]
    OPERATORS = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "mod": operator.mod,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "lshift": operator.lshift,
        "rshift": operator.rshift,
    }
    PLAIN = [0, 1, 2, -3, 6, True, False]
    SYMINT_BODY = ["__bool__", "__int__", "__index__", "__truediv__", "__rtruediv__"]
    SYMINT_BODY += ["__floordiv__", "__rfloordiv__", "__pow__", "__rpow__"]
    SYMINT_BODY += ["__repr__", "__hash__", "has_hint"]
    SYMBOOL_BODY = ["__bool__", "__int__", "__hash__", "__sym_float__", "__repr__"]
    SYMBOOL_BODY += ["__sym_ite__"]
    CORE = {"add", "sub", "mul", "mod", "int_floordiv", "and", "or", "eq", "ne"}
    CORE |= {"lt", "le", "gt", "ge", "neg", "sym_not", "sym_min", "sym_max"}

    def setUp(self):
        super().setUp()
        ShapeEnv(_allow_native=True)
        self.assertTrue(sym_node._native_glue_originals)

    @staticmethod
    def make_env(native=True):
        env = ShapeEnv(_allow_native=native)
        hints = [5, 7, 3, -4, 6]
        syms = [
            env.create_symbol(
                h, ConstantSource(f"x{i}"), DimDynamic.DYNAMIC, positive=h > 0 or None
            )
            for i, h in enumerate(hints)
        ]
        ints = [env.create_symintnode(s, hint=h) for s, h in zip(syms, hints)]
        bools = [
            env.create_symboolnode(sympy.Ne(syms[0], 3)),
            env.create_symboolnode(sympy.Lt(syms[1], syms[2])),
        ]
        return env, ints, bools

    @staticmethod
    def describe(r):
        if isinstance(r, torch.SymInt | torch.SymBool | torch.SymFloat):
            n = r.node
            return (type(r), type(n), str(n), n.hint, n.constant, n.pytype)
        return (type(r), r)

    def program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        out = []
        core = [
            e for e in self.ENTRIES if e[1].strip("_").removeprefix("r") in self.CORE
        ]
        for _ in range(150):
            cls, attr, kind = rng.choice(core if rng.random() < 0.8 else self.ENTRIES)
            pool = ints if cls is torch.SymInt else bools
            x = rng.choice(pool)
            method = attr[3:-2] if kind == "rbinary" else attr[2:-2]
            try:
                if kind == "unary":
                    r = getattr(x, attr)()
                else:
                    if "pow" in method or "shift" in method:
                        y = rng.choice([0, 1, 2])
                    else:
                        y = rng.choice(ints + bools + self.PLAIN)
                    fn = self.OPERATORS.get(method)
                    style = rng.randrange(3)
                    if style == 0 or fn is None:
                        r = getattr(x, attr)(y)
                    elif style == 1:
                        r = getattr(cls, attr)(x, y)
                    elif kind == "binary":
                        r = fn(x, y)
                    else:
                        r = fn(rng.choice(self.PLAIN), x)
            except Exception as e:
                out.append(("raise", type(e)))
                continue
            out.append(self.describe(r))
            if isinstance(r, torch.SymInt):
                target = ints
            elif isinstance(r, torch.SymBool):
                target = bools
            else:
                continue
            if type(r.node) is not _NativeSymNode and rng.random() < 0.8:
                continue
            if len(target) < 12:
                target.append(r)
            else:
                target[rng.randrange(len(target))] = r
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(8))
    def test_differential(self, seed):
        got = self.program(seed)
        with python_glue():
            self.assertIsInstance(torch.SymInt.__add__, types.FunctionType)
            want = self.program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 40)

    def body_program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        ints += [ints[0] + ints[1], ints[2] * 2, ints[3] - 1]
        exponents = [0, 1, 2, -1, True, False]
        ops = [
            lambda x, y, p: bool(x),
            lambda x, y, p: bool(p),
            lambda x, y, p: int(p),
            lambda x, y, p: x / y,
            lambda x, y, p: y / x,
            lambda x, y, p: x // y,
            lambda x, y, p: y // x,
            lambda x, y, p: x ** rng.choice(exponents),
            lambda x, y, p: rng.choice([2, -2, 0]) ** x,
            lambda x, y, p: x ** rng.choice(ints[:5]),
            lambda x, y, p: x.__rpow__(y),
            lambda x, y, p: x / p,
            lambda x, y, p: x // p,
            lambda x, y, p: x**p,
            lambda x, y, p: x / 2.0,
            lambda x, y, p: repr(x),
            lambda x, y, p: repr(p),
            lambda x, y, p: hash(x),
            lambda x, y, p: hash(p),
            lambda x, y, p: x.has_hint(),
            lambda x, y, p: torch.sym_float(p),
            lambda x, y, p: torch.sym_ite(p, x, y),
            lambda x, y, p: torch.sym_ite(p, 3, 3),
            lambda x, y, p: torch.sym_ite(p, rng.choice(bools), rng.choice(bools)),
            lambda x, y, p: p.__sym_ite__(rng.choice(self.PLAIN), y),
            lambda x, y, p: p.__sym_ite__(x, 1.5),
        ]
        out = []
        for _ in range(120):
            x = rng.choice(ints)
            y = rng.choice(ints + self.PLAIN)
            p = rng.choice(bools)
            try:
                r = rng.choice(ops)(x, y, p)
            except Exception as e:
                out.append(("raise", type(e), str(e)))
                continue
            out.append(self.describe(r))
            if isinstance(r, torch.SymInt) and type(r.node) is _NativeSymNode:
                ints[rng.randrange(5, len(ints))] = r
        # int() specializes, so it goes last.
        for x in rng.sample(ints, 3):
            try:
                out.append((int(x), operator.index(x)))
            except Exception as e:
                out.append(("raise", type(e)))
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(6))
    def test_body_differential(self, seed):
        got = self.body_program(seed)
        with python_glue():
            self.assertIsInstance(torch.SymInt.__bool__, types.FunctionType)
            want = self.body_program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 10)

    def test_body_native_path(self):
        originals = sym_node._native_glue_originals.values()
        codes = {fn.__code__ for fn in originals}

        def run():
            _, (a, b, _, d, e), (p, q) = self.make_env()
            calls = []

            def profile(frame, event, arg):
                if event != "call" or frame.f_code not in codes:
                    return
                # SymFloat's magic methods share their code with SymInt's.
                first = frame.f_locals[frame.f_code.co_varnames[0]]
                if not isinstance(first, torch.SymFloat):
                    calls.append(frame.f_code.co_qualname)

            prev = sys.getprofile()
            sys.setprofile(profile)
            try:
                out = [bool(a), bool(p), int(q), a / b, 3 / a, a // 2, 7 // a]
                out += [a**2, a**True, a**b, 2**a, repr(a), repr(q), hash(p)]
                out += [a.has_hint(), torch.sym_ite(p, a, b), torch.sym_ite(q, 1, 2)]
                out += [a**d, 2**d]
                # int() specializes, so it goes last.
                out += [p.__sym_ite__(True, q), int(e), operator.index(e)]
            finally:
                sys.setprofile(prev)
            return [self.describe(r) for r in out], calls

        got, calls = run()
        self.assertEqual(calls, [])
        with python_glue():
            want, calls = run()
        self.assertGreater(len(calls), 20)
        self.assertEqual(got, want)

    def sum_program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        ints += [ints[0] + ints[1], ints[2] * 2, ints[3] - 1]
        py_int = torch.SymInt(SymNode(ints[0].node.expr + 1, env, int, 6))

        class Small(enum.IntEnum):
            ONE = 1

        class Items(list):
            pass

        class Sub(torch.SymInt):
            pass

        odd = [py_int, Small.ONE, 2**70, 1.5, bools[0], Sub(ints[1].node)]
        out = []
        for _ in range(80):
            items = [rng.choice(ints + self.PLAIN) for _ in range(rng.randrange(4))]
            if rng.random() < 0.15:
                items.insert(rng.randrange(len(items) + 1), rng.choice(odd))
            style = rng.randrange(3)
            try:
                if style == 0:
                    r = torch.sym_sum(items if rng.random() < 0.8 else Items(items))
                elif style == 1:
                    r = torch.sym_sum(tuple(items))
                else:
                    r = torch.sym_sum(*items)
                # A bool item reaches SymNode.sym_sum as sympy.true, which sympy.Add rejects.
                out.append(self.describe(r))
            except Exception as e:
                out.append(("raise", type(e), re.sub("0x[0-9a-f]+", "", str(e))))
                continue
            if isinstance(r, torch.SymInt) and type(r.node) is _NativeSymNode:
                ints[rng.randrange(5, len(ints))] = r
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(4))
    def test_sym_sum_differential(self, seed):
        got = self.sum_program(seed)
        with python_glue():
            want = self.sum_program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 20)

    def test_sym_sum_native_path(self):
        codes = {sym_node.to_node.__code__, sym_node.wrap_node.__code__}

        def run():
            _, (a, b, c, *_), _ = self.make_env()
            calls = []

            def profile(frame, event, arg):
                if event == "call" and frame.f_code in codes:
                    calls.append(frame.f_code.co_name)

            prev = sys.getprofile()
            sys.setprofile(profile)
            try:
                out = [torch.sym_sum([a, b, 3]), torch.sym_sum(a, 2, c)]
                out += [torch.sym_sum((4, a)), torch.sym_sum([a, -a])]
            finally:
                sys.setprofile(prev)
            return [self.describe(r) for r in out], calls

        got, calls = run()
        self.assertEqual(calls, [])
        with python_glue():
            want, calls = run()
        self.assertGreater(len(calls), 10)
        self.assertEqual(got, want)

    def test_body_fallbacks(self):
        reasons = []
        sym_float = torch.sym_float

        def recording(x):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return sym_float(x)

        _, (a, b, *_), (p, _) = self.make_env()
        with mock.patch.object(torch, "sym_float", recording):
            self.assertIsInstance(a**-1, torch.SymFloat)
            self.assertIsInstance(a / 2.0, torch.SymFloat)
        self.assertEqual(reasons[0], "negative exponent")
        self.assertEqual(reasons[-1], "operand type")
        with self.assertRaisesRegex(TypeError, "positional argument"):
            torch.SymInt.__pow__(a, 2, 3)
        with self.assertRaisesRegex(TypeError, "non-nested SymInt"):
            hash(a)
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            a / p
        with self.assertRaisesRegex(AssertionError, "same pytype"):
            p.__sym_ite__(a, p)
        unbound = [
            lambda: torch.SymInt.__bool__(p),
            lambda: torch.SymBool.__bool__(a),
            lambda: torch.SymInt.__int__(p),
            lambda: torch.SymBool.__hash__(a),
            lambda: torch.SymInt.__truediv__(p, 2),
            lambda: torch.SymInt.__pow__(p, 2),
            lambda: torch.SymBool.__sym_ite__(a, 1, 2),
        ]
        for fn in unbound:
            results = []
            for ctx in (contextlib.nullcontext(), python_glue()):
                with ctx:
                    try:
                        results.append(self.describe(fn()))
                    except Exception as e:
                        results.append(("raise", type(e), str(e)))
            self.assertEqual(results[0], results[1])
        self.assertIsNone(torch._C._symbolic._glue_fallback_reason())

    def test_installed(self):
        originals = sym_node._native_glue_originals
        body = {(torch.SymInt, a) for a in self.SYMINT_BODY}
        body |= {(torch.SymBool, a) for a in self.SYMBOOL_BODY}
        entries = {(cls, attr) for cls, attr, _ in self.ENTRIES}
        self.assertEqual(set(originals), entries | body)
        for cls, attr in originals:
            glue = cls.__dict__[attr]
            self.assertIsInstance(glue, torch._C._symbolic._SymGlueMethod)
            fn = originals[(cls, attr)]
            self.assertIs(glue.__wrapped__, fn)
            self.assertEqual(glue.__name__, fn.__name__)
            self.assertEqual(glue.__qualname__, fn.__qualname__)
            self.assertEqual(inspect.signature(glue), inspect.signature(fn))
            self.assertIs(getattr(cls, attr), glue)
        self.assertIsInstance(torch.SymFloat.__add__, types.FunctionType)
        self.assertIsInstance(torch.SymInt.__round__, types.FunctionType)
        _, ints, _ = self.make_env()
        a = ints[0]
        self.assertEqual(str(a.__add__(1)), f"{a} + 1")
        self.assertIs(a.__add__.__self__, a)
        self.assertEqual(str(torch.SymInt.__add__(a, 1)), f"{a} + 1")

    def test_native_path(self):
        calls = []

        def counted(fn):
            def wrapper(*args):
                calls.append(fn.__name__)
                return fn(*args)

            return wrapper

        def run(native):
            env, (a, b, c, d, _), (p, q) = self.make_env(native)
            out = [a + b, a * 2, 3 - a, a // b, a % 3, a == 5, a < b, -a, p & q]
            out += [p | True, a > c, 2 * a, torch.sym_min(a, 3), a != 4, a - 5]
            return [self.describe(r) for r in out]

        with (
            mock.patch.object(sym_node, "to_node", counted(sym_node.to_node)),
            mock.patch.object(sym_node, "wrap_node", counted(sym_node.wrap_node)),
        ):
            got = run(True)
            self.assertEqual(calls, [])
            want = run(False)
            self.assertGreater(len(calls), 20)
        for g, w in zip(got, want, strict=True):
            self.assertEqual(g[3:], w[3:])
            assert_same_function(self, sympy.sympify(g[2]), sympy.sympify(w[2]))
        self.assertTrue(all(r[1] is _NativeSymNode for r in got))

    def test_fallbacks(self):
        reasons = []
        to_node = sym_node.to_node

        def recording(self_node, num):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return to_node(self_node, num)

        class Small(enum.IntEnum):
            ONE = 1

        cases = [
            (lambda a, p: a + 1.5, "operand type"),
            (lambda a, p: a * Small.ONE, "operand type"),
            (lambda a, p: a + 2**70, "int beyond int64"),
            (lambda a, p: torch.SymInt.__add__(a, other=1), "keywords"),
            (lambda a, p: torch.SymInt.__sub__(a, 1, 2), "arity"),
            (lambda a, p: torch.SymInt.__neg__(a, 1), "arity"),
            (
                lambda a, p: a
                + torch.SymInt(SymNode(a.node._expr, a.node.shape_env, int, 5)),
                "python node",
            ),
            (lambda a, p: a + "x", "operand type"),
            (lambda a, p: 1.5 - a, "operand type"),
            (lambda a, p: p & 1.5, "operand type"),
            (lambda a, p: torch.SymInt.__add__(3, 4), "operand type"),
        ]
        for fn, reason in cases:
            results = []
            for glue in (True, False):
                _, ints, bools = self.make_env()
                reasons.clear()
                ctx = contextlib.nullcontext() if glue else python_glue()
                with ctx, mock.patch.object(sym_node, "to_node", recording):
                    try:
                        results.append(self.describe(fn(ints[0], bools[0])))
                    except Exception as e:
                        results.append(("raise", type(e)))
                if glue and reasons:
                    self.assertEqual(reasons[0], reason)
            self.assertEqual(results[0], results[1])
        self.assertIsNone(torch._C._symbolic._glue_fallback_reason())

    def test_magic_logging(self):
        _, (a, *_), _ = self.make_env()
        with self.assertLogs(sym_node.sym_node_log, level="DEBUG") as logs:
            r = a + 1
        self.assertEqual(
            logs.output, [f"DEBUG:{sym_node.sym_node_log.name}:MAGIC add {a} 1"]
        )
        self.assertIsInstance(r.node, _NativeSymNode)

    def test_modified_class(self):
        _, (a, *_), _ = self.make_env()
        reasons = []
        to_node = sym_node.to_node

        def recording(self_node, num):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return to_node(self_node, num)

        with mock.patch.object(sym_node, "to_node", recording):
            try:
                torch.SymInt._glue_test_attr = 1
                self.assertEqual(str(a + 1), f"{a} + 1")
                del torch.SymInt._glue_test_attr
                self.assertEqual(str(a + 1), f"{a} + 1")
                self.assertEqual(reasons, ["glue modified"] * 2)
            finally:
                sym_node._install_native_glue()
            reasons.clear()
            self.assertEqual(str(a + 1), f"{a} + 1")
            self.assertEqual(reasons, [])

    @parametrize("pre_dispatch", [False, True])
    def test_make_fx(self, pre_dispatch):
        def f(a, b, p, d):
            c = a + b
            d = c * 2 - a // b
            e = (d % 3 + 1) * 1
            m = torch.sym_max(a, b) + 3 * torch.sym_min(a, 3)
            r = e, m, -a, 2 - a, a < b, (p & (a == b)) | (b > 2), p + 1, True * p
            r += a**2, a**b, 2**a, a // 2, 7 // b, a / b, torch.sym_ite(p, a, b)
            return *r, a**d, 2**d, torch.sym_sum([a, b, 3]), torch.sym_sum(4, a)

        def trace():
            env, (a, b, _, d, _), (p, _) = self.make_env()
            fx = make_fx(f, tracing_mode="real", pre_dispatch=pre_dispatch)
            with self.count_inits() as inits:
                gm = fx(a, b, p, d)
            return gm.code, [str(g.expr) for g in env.guards], len(inits)

        *got, got_inits = trace()
        with python_glue():
            *want, want_inits = trace()
        self.assertEqual(got, want)
        self.assertLess(got_inits, want_inits)

    @staticmethod
    @contextlib.contextmanager
    def count_inits():
        codes = {torch.SymInt.__init__.__code__, torch.SymBool.__init__.__code__}
        inits = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code in codes:
                inits.append(frame.f_code.co_qualname)

        prev = sys.getprofile()
        sys.setprofile(profile)
        try:
            yield inits
        finally:
            sys.setprofile(prev)

    def test_native_construction(self):
        _, (a, b, *_), (p, _) = self.make_env()
        t = torch.empty((a, b), device="meta")
        with self.count_inits() as inits:
            size, stride = t.size(0), t.stride(0)
            numel = t.numel()
            q = torch._C._symbolic._roundtrip_symbool(p)
        self.assertEqual(inits, [])
        self.assertIs(type(size), torch.SymInt)
        self.assertEqual(size.__dict__, {"node": a.node})
        self.assertEqual(str(stride), str(b))
        self.assertEqual(str(numel), str(a * b))
        self.assertIs(type(q), torch.SymBool)
        self.assertEqual(q.__dict__, {"node": p.node})

        calls = []
        init = torch.SymInt.__init__

        def counted(self, node):
            calls.append(node)
            init(self, node)

        try:
            with mock.patch.object(torch.SymInt, "__init__", counted):
                self.assertIs(t.size(0).node, a.node)
            self.assertEqual(calls, [a.node])
        finally:
            sym_node._install_native_glue()
        with self.count_inits() as inits:
            t.size(0)
        self.assertEqual(inits, [])

        init = torch.SymBool.__init__
        try:
            with mock.patch.object(torch.SymBool, "__init__", counted):
                sym_node._install_native_glue()
                self.assertIs(
                    type(torch._C._symbolic._roundtrip_symbool(p)), torch.SymBool
                )
            self.assertEqual(calls, [a.node, p.node])
        finally:
            sym_node._install_native_glue()
        with self.count_inits() as inits:
            torch._C._symbolic._roundtrip_symbool(p)
        self.assertEqual(inits, [])

    def test_flag_off_unchanged(self):
        script = (
            "import types, torch\n"
            "from torch.fx.experimental import sym_node\n"
            "from torch.fx.experimental.symbolic_shapes import ShapeEnv\n"
            "ShapeEnv()\n"
            "print(isinstance(torch.SymInt.__add__, types.FunctionType),"
            " not sym_node._native_glue_originals)\n"
        )
        env = dict(os.environ)
        env.pop("CPP_SYMNODE", None)
        out = subprocess.check_output(
            [sys.executable, "-c", script], env=env, text=True
        )
        self.assertEqual(out.strip().splitlines()[-1], "True True")


instantiate_parametrized_tests(TestNativeExpr)
instantiate_parametrized_tests(TestNativeShapeEnv)
instantiate_parametrized_tests(TestNativeShapeEnvSync)
instantiate_parametrized_tests(TestNativeSymNode)
instantiate_parametrized_tests(TestNativeSymNodeCompile)
instantiate_parametrized_tests(TestNativeSymIntGlue)


if __name__ == "__main__":
    run_tests()
