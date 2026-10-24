#include <torch/csrc/symbolic/ValueRanges.h>

#include <torch/csrc/symbolic/Functions.h>

#include <c10/util/Exception.h>

#include <symengine/add.h>
#include <symengine/infinity.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/pow.h>

#include <algorithm>
#include <array>

namespace torch::symbolic {

using SymEngine::is_a;

namespace {

bool is_int_bound(const Expr& e) {
  return is_integer(e) || is_int_oo(e);
}

int sign(const Expr& b) {
  return compare_numbers(b, Expr(0));
}

const Expr& bmin(const Expr& a, const Expr& b) {
  return compare_numbers(a, b) <= 0 ? a : b;
}

const Expr& bmax(const Expr& a, const Expr& b) {
  return compare_numbers(a, b) >= 0 ? a : b;
}

Expr badd(const Expr& a, const Expr& b) {
  if (is_int_oo(a) && is_int_oo(b) && a != b) {
    throw NativeUnsupported("int_oo - int_oo");
  }
  if (is_int_oo(a)) {
    return a;
  }
  if (is_int_oo(b)) {
    return b;
  }
  return a + b;
}

// 0 * int_oo is 0, as ValueRanges' safe_mul has it.
Expr bmul(const Expr& a, const Expr& b) {
  int s = sign(a) * sign(b);
  if (s == 0) {
    return Expr(0);
  }
  if (is_int_oo(a) || is_int_oo(b)) {
    return s > 0 ? int_oo() : neg_int_oo();
  }
  return a * b;
}

// b ** n for an int64 n >= 0.
Expr bpow(const Expr& b, int64_t n) {
  if (n == 0) {
    return Expr(1);
  }
  if (is_int_oo(b)) {
    return sign(b) > 0 || n % 2 == 0 ? int_oo() : neg_int_oo();
  }
  if (n > 63 && compare_numbers(bmax(b, -b), Expr(1)) > 0) {
    throw NativeUnsupported("power bound overflow");
  }
  return SymEngine::pow(b, Expr(n));
}

// floor(x / y) for y > 0.
Expr bfloordiv(const Expr& x, const Expr& y) {
  if (is_int_oo(x)) {
    return x;
  }
  if (is_int_oo(y)) {
    return Expr(sign(x) >= 0 ? 0 : -1);
  }
  return function(Fn::FloorDiv, {x, y});
}

class Analysis {
 public:
  Analysis(const RangeMap& ranges, const SymbolTable& symbols)
      : ranges_(&ranges), symbols_(&symbols) {}

  ValueRanges interp(const Expr& e);

 private:
  ValueRanges int_range(const Expr& e) {
    ValueRanges r = interp(e);
    if (!r.is_int()) {
      throw NativeUnsupported("expected an integer range");
    }
    return r;
  }
  ValueRanges bool_range(const Expr& e) {
    ValueRanges r = interp(e);
    if (!r.is_bool()) {
      throw NativeUnsupported("expected a boolean range");
    }
    return r;
  }

  ValueRanges symbol(const Expr& s);
  ValueRanges mul(const ValueRanges& x, const ValueRanges& y);
  ValueRanges pow(const ValueRanges& base, int64_t n);
  ValueRanges floordiv(const ValueRanges& x, const ValueRanges& y);
  ValueRanges mod(const ValueRanges& x, const ValueRanges& y);
  ValueRanges pow_by_natural(const ValueRanges& base, const ValueRanges& exp);
  ValueRanges relational(const Expr& e);
  ValueRanges function(const TorchFunction& f, const Expr& e);

  const RangeMap* ranges_;
  const SymbolTable* symbols_;
};

ValueRanges Analysis::symbol(const Expr& s) {
  auto it = ranges_->find(s);
  if (it != ranges_->end()) {
    return it->second;
  }
  const SymbolFacts* f = symbols_->facts(s);
  if (f == nullptr || !f->integer) {
    throw NativeUnsupported("symbol without a range: " + symbol_name(s));
  }
  if (f->positive) {
    return {Expr(1), int_oo()};
  }
  if (f->nonnegative) {
    return {Expr(0), int_oo()};
  }
  return {neg_int_oo(), int_oo()};
}

ValueRanges Analysis::mul(const ValueRanges& x, const ValueRanges& y) {
  std::array<Expr, 4> c = {
      bmul(x.lower, y.lower),
      bmul(x.lower, y.upper),
      bmul(x.upper, y.lower),
      bmul(x.upper, y.upper)};
  Expr lo = c[0];
  Expr hi = c[0];
  for (const Expr& v : c) {
    lo = bmin(lo, v);
    hi = bmax(hi, v);
  }
  return {lo, hi};
}

ValueRanges Analysis::pow(const ValueRanges& base, int64_t n) {
  Expr pl = bpow(base.lower, n);
  Expr pu = bpow(base.upper, n);
  if (n % 2 == 1 || sign(base.lower) >= 0) {
    return {pl, pu};
  }
  if (sign(base.upper) <= 0) {
    return {pu, pl};
  }
  return {Expr(0), bmax(pl, pu)};
}

ValueRanges Analysis::floordiv(const ValueRanges& x, const ValueRanges& y) {
  if (sign(y.lower) <= 0) {
    throw NativeUnsupported("FloorDiv by a range that is not positive");
  }
  // floor(x / y) increases with x; for y > 0 it decreases with y when x >= 0
  // and increases with y when x < 0.
  Expr lo = bfloordiv(x.lower, sign(x.lower) >= 0 ? y.upper : y.lower);
  Expr hi = bfloordiv(x.upper, sign(x.upper) >= 0 ? y.lower : y.upper);
  return {lo, hi};
}

// Mod and PythonMod with y > 0, which agree for x >= 0.
ValueRanges Analysis::mod(const ValueRanges& x, const ValueRanges& y) {
  if (sign(y.lower) <= 0) {
    throw NativeUnsupported("Mod by a range that is not positive");
  }
  Expr hi = badd(y.upper, Expr(-1));
  if (sign(x.lower) >= 0) {
    if (compare_numbers(x.upper, y.lower) < 0) {
      return x;
    }
    hi = bmin(hi, x.upper);
  }
  return {Expr(0), hi};
}

ValueRanges Analysis::pow_by_natural(
    const ValueRanges& base,
    const ValueRanges& exp) {
  if (exp.is_singleton()) {
    auto n = as_int64(exp.lower);
    if (!n || *n < 0) {
      throw NativeUnsupported("PowByNatural exponent");
    }
    return pow(base, *n);
  }
  // Monotone in both arguments for base >= 1 and exp >= 0.
  if (compare_numbers(base.lower, Expr(1)) < 0 || sign(exp.lower) < 0) {
    throw NativeUnsupported("PowByNatural range");
  }
  auto at = [](const Expr& b, const Expr& e) {
    if (is_int_oo(e)) {
      return b == Expr(1) ? Expr(1) : int_oo();
    }
    auto n = as_int64(e);
    if (!n) {
      throw NativeUnsupported("PowByNatural exponent");
    }
    return bpow(b, *n);
  };
  return {at(base.lower, exp.lower), at(base.upper, exp.upper)};
}

ValueRanges Analysis::relational(const Expr& e) {
  auto a = args(e);
  if (is_boolean(a[0]) || is_boolean(a[1])) {
    throw NativeUnsupported("relational of booleans");
  }
  // lhs OP rhs as a statement about d = rhs - lhs, which cancels the terms
  // the two sides share.
  ValueRanges d = int_range(expand(a[1] - a[0]));
  int lo = sign(d.lower);
  int hi = sign(d.upper);
  const SymEngine::Basic& b = *e.get_basic();
  if (is_a<SymEngine::StrictLessThan>(b)) {
    return ValueRanges::boolean_range(lo <= 0, hi > 0);
  }
  if (is_a<SymEngine::LessThan>(b)) {
    return ValueRanges::boolean_range(lo < 0, hi >= 0);
  }
  bool can_be_eq = lo <= 0 && hi >= 0;
  bool must_be_eq = lo == 0 && hi == 0;
  if (is_a<SymEngine::Equality>(b)) {
    return ValueRanges::boolean_range(!must_be_eq, can_be_eq);
  }
  return ValueRanges::boolean_range(can_be_eq, !must_be_eq);
}

ValueRanges Analysis::function(const TorchFunction& f, const Expr& e) {
  auto a = args(e);
  switch (f.fn()) {
    case Fn::FloorDiv:
    case Fn::CleanDiv:
      return floordiv(int_range(a[0]), int_range(a[1]));
    case Fn::Mod: {
      ValueRanges x = int_range(a[0]);
      if (sign(x.lower) < 0) {
        throw NativeUnsupported("Mod of a range that is not nonnegative");
      }
      return mod(x, int_range(a[1]));
    }
    case Fn::PythonMod:
      return mod(int_range(a[0]), int_range(a[1]));
    case Fn::PowByNatural:
      return pow_by_natural(int_range(a[0]), int_range(a[1]));
    case Fn::IsNonOverlappingAndDenseIndicator:
      return {Expr(0), Expr(1)};
  }
  TORCH_INTERNAL_ASSERT(false);
}

ValueRanges Analysis::interp(const Expr& e) {
  const SymEngine::Basic& b = *e.get_basic();
  if (is_integer(e)) {
    return {e, e};
  }
  if (as_bool(e)) {
    return {e, e};
  }
  if (is_symbol(e)) {
    return symbol(e);
  }
  if (is_a<SymEngine::Add>(b)) {
    auto a = args(e);
    ValueRanges r = int_range(a[0]);
    for (size_t i = 1; i < a.size(); ++i) {
      ValueRanges t = int_range(a[i]);
      r = {badd(r.lower, t.lower), badd(r.upper, t.upper)};
    }
    return r;
  }
  if (is_a<SymEngine::Mul>(b)) {
    auto a = args(e);
    ValueRanges r = int_range(a[0]);
    for (size_t i = 1; i < a.size(); ++i) {
      r = mul(r, int_range(a[i]));
    }
    return r;
  }
  if (is_a<SymEngine::Pow>(b)) {
    auto a = args(e);
    auto n = as_int64(a[1]);
    if (!n || *n < 0) {
      throw NativeUnsupported("Pow with an exponent that is not natural");
    }
    return pow(int_range(a[0]), *n);
  }
  if (is_max(e) || is_min(e)) {
    bool max = is_max(e);
    auto a = args(e);
    ValueRanges r = int_range(a[0]);
    for (size_t i = 1; i < a.size(); ++i) {
      ValueRanges t = int_range(a[i]);
      r = max ? ValueRanges(bmax(r.lower, t.lower), bmax(r.upper, t.upper))
              : ValueRanges(bmin(r.lower, t.lower), bmin(r.upper, t.upper));
    }
    return r;
  }
  if (const TorchFunction* f = as_function(e)) {
    return function(*f, e);
  }
  if (is_relational(e)) {
    return relational(e);
  }
  if (is_a<SymEngine::Not>(b)) {
    ValueRanges x = bool_range(args(e)[0]);
    return ValueRanges::boolean_range(x.can_be_true(), x.can_be_false());
  }
  if (is_a<SymEngine::And>(b) || is_a<SymEngine::Or>(b)) {
    bool all_true = true;
    bool any_true = false;
    bool all_false = true;
    bool any_false = false;
    for (const Expr& a : args(e)) {
      ValueRanges x = bool_range(a);
      all_true = all_true && x.can_be_true();
      any_true = any_true || x.can_be_true();
      all_false = all_false && x.can_be_false();
      any_false = any_false || x.can_be_false();
    }
    return is_a<SymEngine::And>(b)
        ? ValueRanges::boolean_range(any_false, all_true)
        : ValueRanges::boolean_range(all_false, any_true);
  }
  throw NativeUnsupported("no value range for this expression");
}

} // namespace

ValueRanges::ValueRanges(Expr lower_, Expr upper_)
    : lower(std::move(lower_)), upper(std::move(upper_)) {
  auto lower_bool = as_bool(lower);
  auto upper_bool = as_bool(upper);
  if (lower_bool && upper_bool) {
    TORCH_CHECK(
        *lower_bool <= *upper_bool, "invalid range [", lower, ", ", upper, "]");
    return;
  }
  TORCH_CHECK(
      is_int_bound(lower) && is_int_bound(upper),
      "range bounds must be integers or int_oo: [",
      lower,
      ", ",
      upper,
      "]");
  TORCH_CHECK(
      compare_numbers(lower, upper) <= 0,
      "invalid range [",
      lower,
      ", ",
      upper,
      "]");
}

ValueRanges ValueRanges::boolean_range(bool can_be_false, bool can_be_true) {
  TORCH_INTERNAL_ASSERT(can_be_false || can_be_true);
  return {boolean(!can_be_false), boolean(can_be_true)};
}

bool ValueRanges::is_bool() const {
  return as_bool(lower).has_value();
}

bool ValueRanges::is_int() const {
  return !is_bool();
}

ValueRanges bound_sympy(
    const Expr& e,
    const RangeMap& ranges,
    const SymbolTable& symbols) {
  return Analysis(ranges, symbols).interp(e);
}

} // namespace torch::symbolic
