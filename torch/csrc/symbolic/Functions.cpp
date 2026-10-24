#include <torch/csrc/symbolic/Functions.h>

#include <c10/util/Exception.h>

#include <symengine/add.h>
#include <symengine/integer.h>
#include <symengine/pow.h>

#include <algorithm>
#include <array>
#include <numeric>

namespace torch::symbolic {

using SymEngine::Basic;
using SymEngine::integer_class;
using SymEngine::RCP;

namespace {

constexpr std::array kNames = {
    "FloorDiv",
    "CleanDiv",
    "Mod",
    "PythonMod",
    "PowByNatural",
    "IsNonOverlappingAndDenseIndicator",
};

const integer_class& as_mp(const Expr& e) {
  return SymEngine::down_cast<const SymEngine::Integer&>(*e.get_basic())
      .as_integer_class();
}

Expr from_mp(const integer_class& i) {
  return Expr(SymEngine::integer(i));
}

std::vector<Expr> terms_of(const Expr& e) {
  if (SymEngine::is_a<SymEngine::Add>(*e.get_basic())) {
    return args(e);
  }
  return {e};
}

// Whether term is an integer multiple of the nonzero integer q.
bool multiple_of(const Expr& term, const integer_class& q) {
  auto [coeff, rest] = as_coeff_mul(term);
  if (!is_integer(coeff)) {
    return false;
  }
  integer_class r;
  SymEngine::mp_fdiv_r(r, as_mp(coeff), q);
  return r == 0;
}

Expr eval_floordiv(Fn fn, const Expr& base, const Expr& divisor) {
  if (divisor == Expr(0)) {
    throw NativeUnsupported("division by zero");
  }
  if (is_int_oo(base) || is_int_oo(divisor)) {
    throw NativeUnsupported("FloorDiv of int_oo");
  }
  if (base == Expr(0)) {
    return base;
  }
  if (divisor == Expr(1)) {
    return base;
  }
  if (divisor == Expr(-1)) {
    return -base;
  }
  if (base == divisor) {
    return Expr(1);
  }
  if (base == -divisor) {
    return Expr(-1);
  }
  if (is_integer(base) && is_integer(divisor)) {
    integer_class q;
    SymEngine::mp_fdiv_q(q, as_mp(base), as_mp(divisor));
    return from_mp(q);
  }
  if (const TorchFunction* f = as_function(base);
      f != nullptr && (f->fn() == Fn::FloorDiv || f->fn() == Fn::CleanDiv)) {
    auto a = args(base);
    return function(fn, {a[0], a[1] * divisor});
  }
  if (!is_integer(divisor)) {
    return Expr(SymEngine::make_rcp<const TorchFunction>(
        fn, SymEngine::vec_basic{base.get_basic(), divisor.get_basic()}));
  }
  // (q*x + y) // q == x + y // q for integer x.
  const integer_class& d = as_mp(divisor);
  std::vector<Expr> quotients;
  std::vector<Expr> rest;
  for (const Expr& t : terms_of(expand(base))) {
    if (multiple_of(t, d)) {
      quotients.push_back(t / divisor);
    } else {
      rest.push_back(t);
    }
  }
  if (!quotients.empty()) {
    return add(quotients) + function(fn, {add(rest), divisor});
  }
  // (g*x) // (g*q) == x // q for g > 0.
  integer_class g = d;
  for (const Expr& t : rest) {
    auto [coeff, r] = as_coeff_mul(t);
    if (!is_integer(coeff)) {
      g = 1;
      break;
    }
    SymEngine::mp_gcd(g, g, as_mp(coeff));
  }
  if (g < 0) {
    g = -g;
  }
  if (g > 1) {
    Expr ge = from_mp(g);
    return function(fn, {expand(base / ge), divisor / ge});
  }
  return Expr(SymEngine::make_rcp<const TorchFunction>(
      fn, SymEngine::vec_basic{base.get_basic(), divisor.get_basic()}));
}

// Mod (for p >= 0, q > 0) and PythonMod.
Expr eval_mod(Fn fn, const Expr& p, const Expr& q) {
  if (q == Expr(0)) {
    throw NativeUnsupported("modulo by zero");
  }
  if (is_int_oo(p) || is_int_oo(q)) {
    throw NativeUnsupported("Mod of int_oo");
  }
  if (p == Expr(0) || p == q || p == -q || q == Expr(1)) {
    return Expr(0);
  }
  if (is_integer(p) && is_integer(q)) {
    if (fn == Fn::Mod && (as_mp(p) < 0 || as_mp(q) < 1)) {
      throw NativeUnsupported("Mod of a negative number");
    }
    integer_class r;
    SymEngine::mp_fdiv_r(r, as_mp(p), as_mp(q));
    return from_mp(r);
  }
  Expr unevaluated(SymEngine::make_rcp<const TorchFunction>(
      fn, SymEngine::vec_basic{p.get_basic(), q.get_basic()}));
  if (!is_integer(q)) {
    return unevaluated;
  }
  // (q*x + y) % q == y % q for integer x. Mod is only defined for y >= 0,
  // so it drops terms only when all of them go.
  std::vector<Expr> terms = terms_of(expand(p));
  std::vector<Expr> rest;
  for (const Expr& t : terms) {
    if (!multiple_of(t, as_mp(q))) {
      rest.push_back(t);
    }
  }
  if (rest.empty()) {
    return Expr(0);
  }
  if (fn == Fn::Mod || rest.size() == terms.size()) {
    return unevaluated;
  }
  return function(fn, {add(rest), q});
}

Expr eval_pow_by_natural(const Expr& base, const Expr& exp) {
  if (is_integer(exp)) {
    const integer_class& e = as_mp(exp);
    if (e < 0) {
      throw NativeUnsupported("PowByNatural with a negative exponent");
    }
    if (is_integer(base)) {
      const integer_class& b = as_mp(base);
      // Results that do not fit an int64 are left to Python.
      if (b * b > 1 && e > 63) {
        throw NativeUnsupported("PowByNatural overflow");
      }
      Expr r = SymEngine::pow(base, exp);
      if (!as_int64(r)) {
        throw NativeUnsupported("PowByNatural overflow");
      }
      return r;
    }
    return SymEngine::pow(base, exp);
  }
  if (is_int_oo(exp)) {
    throw NativeUnsupported("PowByNatural to int_oo");
  }
  return Expr(SymEngine::make_rcp<const TorchFunction>(
      Fn::PowByNatural,
      SymEngine::vec_basic{base.get_basic(), exp.get_basic()}));
}

Expr eval_indicator(c10::ArrayRef<Expr> args) {
  TORCH_CHECK(args.size() % 2 == 0, "indicator needs sizes and strides");
  std::vector<int64_t> ints;
  for (const Expr& a : args) {
    auto v = as_int64(a);
    if (!v) {
      return Expr(SymEngine::make_rcp<const TorchFunction>(
          Fn::IsNonOverlappingAndDenseIndicator,
          SymEngine::vec_basic(args.begin(), args.end())));
    }
    ints.push_back(*v);
  }
  size_t dim = ints.size() / 2;
  c10::ArrayRef<int64_t> all(ints);
  return Expr(
      eval_is_non_overlapping_and_dense(all.slice(0, dim), all.slice(dim)));
}

} // namespace

const char* function_name(Fn fn) {
  return kNames[static_cast<size_t>(fn)];
}

std::optional<Fn> function_from_name(const std::string& name) {
  for (size_t i = 0; i < std::size(kNames); ++i) {
    if (name == kNames[i]) {
      return static_cast<Fn>(i);
    }
  }
  return std::nullopt;
}

TorchFunction::TorchFunction(Fn fn, const SymEngine::vec_basic& args)
    : FunctionWrapper(function_name(fn), args), fn_(fn) {}

RCP<const Basic> TorchFunction::create(const SymEngine::vec_basic& v) const {
  std::vector<Expr> a(v.begin(), v.end());
  return function(fn_, a).get_basic();
}

RCP<const SymEngine::Number> TorchFunction::eval(long /*bits*/) const {
  throw NativeUnsupported(
      std::string("numeric evaluation of ") + function_name(fn_));
}

RCP<const Basic> TorchFunction::diff_impl(
    const RCP<const SymEngine::Symbol>& /*s*/) const {
  throw NativeUnsupported(std::string("derivative of ") + function_name(fn_));
}

bool TorchFunction::__eq__(const Basic& o) const {
  const auto* f = dynamic_cast<const TorchFunction*>(&o);
  return f != nullptr && f->fn_ == fn_ &&
      SymEngine::unified_eq(get_vec(), f->get_vec());
}

int TorchFunction::compare(const Basic& o) const {
  const auto& f = dynamic_cast<const TorchFunction&>(o);
  if (fn_ != f.fn_) {
    return fn_ < f.fn_ ? -1 : 1;
  }
  return SymEngine::unified_compare(get_vec(), f.get_vec());
}

const TorchFunction* as_function(const Expr& e) {
  if (!SymEngine::is_a<SymEngine::FunctionWrapper>(*e.get_basic())) {
    return nullptr;
  }
  return dynamic_cast<const TorchFunction*>(e.get_basic().get());
}

bool is_function(const Expr& e, Fn fn) {
  const TorchFunction* f = as_function(e);
  return f != nullptr && f->fn() == fn;
}

Expr function(Fn fn, c10::ArrayRef<Expr> args) {
  switch (fn) {
    case Fn::FloorDiv:
    case Fn::CleanDiv:
      TORCH_CHECK(args.size() == 2, function_name(fn), " takes 2 args");
      return eval_floordiv(fn, args[0], args[1]);
    case Fn::Mod:
    case Fn::PythonMod:
      TORCH_CHECK(args.size() == 2, function_name(fn), " takes 2 args");
      return eval_mod(fn, args[0], args[1]);
    case Fn::PowByNatural:
      TORCH_CHECK(args.size() == 2, "PowByNatural takes 2 args");
      return eval_pow_by_natural(args[0], args[1]);
    case Fn::IsNonOverlappingAndDenseIndicator:
      return eval_indicator(args);
  }
  TORCH_INTERNAL_ASSERT(false);
}

Expr sym_max(c10::ArrayRef<Expr> args) {
  SymEngine::vec_basic v(args.begin(), args.end());
  return Expr(SymEngine::max(v));
}

Expr sym_min(c10::ArrayRef<Expr> args) {
  SymEngine::vec_basic v(args.begin(), args.end());
  return Expr(SymEngine::min(v));
}

bool is_max(const Expr& e) {
  return SymEngine::is_a<SymEngine::Max>(*e.get_basic());
}

bool is_min(const Expr& e) {
  return SymEngine::is_a<SymEngine::Min>(*e.get_basic());
}

int64_t eval_is_non_overlapping_and_dense(
    c10::ArrayRef<int64_t> sizes,
    c10::ArrayRef<int64_t> strides) {
  if (sizes.size() == 1) {
    return strides[0] == 1 || sizes[0] < 2;
  }
  std::vector<size_t> order(sizes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return strides[a] < strides[b];
  });
  int64_t expected = 1;
  for (size_t i : order) {
    if (sizes[i] == 1) {
      continue;
    }
    if (strides[i] != expected) {
      return 0;
    }
    expected *= sizes[i];
  }
  return 1;
}

} // namespace torch::symbolic
