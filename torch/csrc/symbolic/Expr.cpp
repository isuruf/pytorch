#include <torch/csrc/symbolic/Expr.h>

#include <torch/csrc/symbolic/Functions.h>

#include <c10/util/Exception.h>
#include <c10/util/hash.h>

#include <symengine/add.h>
#include <symengine/infinity.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/subs.h>
#include <symengine/visitor.h>

#include <algorithm>

namespace torch::symbolic {

using SymEngine::Basic;
using SymEngine::down_cast;
using SymEngine::is_a;
using SymEngine::RCP;

namespace {

SymEngine::vec_basic to_vec(c10::ArrayRef<Expr> args) {
  SymEngine::vec_basic v;
  v.reserve(args.size());
  for (const Expr& a : args) {
    v.push_back(a.get_basic());
  }
  return v;
}

RCP<const SymEngine::Boolean> as_boolean(const Expr& e) {
  TORCH_CHECK(is_boolean(e), "expected a boolean expression, got ", e);
  return SymEngine::rcp_static_cast<const SymEngine::Boolean>(e.get_basic());
}

SymEngine::set_boolean to_set(c10::ArrayRef<Expr> args) {
  SymEngine::set_boolean s;
  for (const Expr& a : args) {
    s.insert(as_boolean(a));
  }
  return s;
}

// -1, 0 or 1 for -int_oo, a finite number and int_oo.
int infinity_rank(const Expr& e) {
  if (!is_int_oo(e)) {
    return 0;
  }
  return down_cast<const SymEngine::Infty&>(*e.get_basic()).is_positive() ? 1
                                                                          : -1;
}

bool decide(Rel kind, int cmp) {
  switch (kind) {
    case Rel::Eq:
      return cmp == 0;
    case Rel::Ne:
      return cmp != 0;
    case Rel::Lt:
      return cmp < 0;
    case Rel::Le:
      return cmp <= 0;
    case Rel::Gt:
      return cmp > 0;
    case Rel::Ge:
      return cmp >= 0;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace

size_t ExprPairHash::operator()(const std::pair<Expr, Expr>& p) const {
  return c10::hash_combine(ExprHash()(p.first), ExprHash()(p.second));
}

Expr int_oo() {
  return Expr(SymEngine::Inf);
}

Expr neg_int_oo() {
  return Expr(SymEngine::NegInf);
}

Expr boolean(bool v) {
  return Expr(SymEngine::boolean(v));
}

bool is_int_oo(const Expr& e) {
  return is_a<SymEngine::Infty>(*e.get_basic());
}

bool is_integer(const Expr& e) {
  return is_a<SymEngine::Integer>(*e.get_basic());
}

bool is_symbol(const Expr& e) {
  return is_a<SymEngine::Symbol>(*e.get_basic());
}

bool is_number(const Expr& e) {
  return SymEngine::is_a_Number(*e.get_basic());
}

bool is_boolean(const Expr& e) {
  return SymEngine::is_a_Boolean(*e.get_basic());
}

bool is_relational(const Expr& e) {
  return SymEngine::is_a_Relational(*e.get_basic());
}

std::optional<bool> as_bool(const Expr& e) {
  if (!is_a<SymEngine::BooleanAtom>(*e.get_basic())) {
    return std::nullopt;
  }
  return down_cast<const SymEngine::BooleanAtom&>(*e.get_basic()).get_val();
}

std::optional<int64_t> as_int64(const Expr& e) {
  if (!is_integer(e)) {
    return std::nullopt;
  }
  const auto& i =
      down_cast<const SymEngine::Integer&>(*e.get_basic()).as_integer_class();
  if (!SymEngine::mp_fits_slong_p(i)) {
    return std::nullopt;
  }
  return SymEngine::mp_get_si(i);
}

const std::string& symbol_name(const Expr& e) {
  TORCH_CHECK(is_symbol(e), "expected a symbol, got ", e);
  return down_cast<const SymEngine::Symbol&>(*e.get_basic()).get_name();
}

std::vector<Expr> args(const Expr& e) {
  std::vector<Expr> r;
  for (const auto& a : e.get_basic()->get_args()) {
    r.emplace_back(a);
  }
  return r;
}

std::vector<Expr> free_symbols(const Expr& e) {
  std::vector<Expr> r;
  for (const auto& s : SymEngine::free_symbols(*e.get_basic())) {
    r.emplace_back(s);
  }
  return r;
}

Expr xreplace(const Expr& e, c10::ArrayRef<std::pair<Expr, Expr>> reps) {
  SymEngine::map_basic_basic m;
  for (const auto& [old, rep] : reps) {
    m[old.get_basic()] = rep.get_basic();
  }
  return Expr(SymEngine::xreplace(e.get_basic(), m));
}

Expr add(c10::ArrayRef<Expr> args) {
  return Expr(SymEngine::add(to_vec(args)));
}

Expr mul(c10::ArrayRef<Expr> args) {
  return Expr(SymEngine::mul(to_vec(args)));
}

std::pair<Expr, Expr> as_coeff_mul(const Expr& e) {
  if (is_number(e)) {
    return {e, Expr(1)};
  }
  if (is_a<SymEngine::Mul>(*e.get_basic())) {
    Expr coeff(down_cast<const SymEngine::Mul&>(*e.get_basic()).get_coef());
    return {coeff, e / coeff};
  }
  return {Expr(1), e};
}

int compare_numbers(const Expr& a, const Expr& b) {
  TORCH_CHECK(is_number(a) && is_number(b), "expected numbers: ", a, ", ", b);
  int ra = infinity_rank(a);
  int rb = infinity_rank(b);
  if (ra != 0 || rb != 0) {
    return ra == rb ? 0 : (ra < rb ? -1 : 1);
  }
  Expr diff = a - b;
  const auto& d = down_cast<const SymEngine::Number&>(*diff.get_basic());
  return d.is_positive() ? 1 : (d.is_zero() ? 0 : -1);
}

Expr rel(Rel kind, const Expr& lhs, const Expr& rhs, bool evaluate) {
  if (evaluate) {
    if (is_number(lhs) && is_number(rhs)) {
      return boolean(decide(kind, compare_numbers(lhs, rhs)));
    }
    if (lhs == rhs) {
      return boolean(decide(kind, 0));
    }
    if (!is_int_oo(lhs) && !is_int_oo(rhs)) {
      Expr d = expand(lhs - rhs);
      if (is_number(d)) {
        return boolean(decide(kind, compare_numbers(d, Expr(0))));
      }
    }
  }
  const auto& l = lhs.get_basic();
  const auto& r = rhs.get_basic();
  switch (kind) {
    case Rel::Eq:
      return Expr(SymEngine::make_rcp<const SymEngine::Equality>(l, r));
    case Rel::Ne:
      return Expr(SymEngine::make_rcp<const SymEngine::Unequality>(l, r));
    case Rel::Lt:
      return Expr(SymEngine::make_rcp<const SymEngine::StrictLessThan>(l, r));
    case Rel::Le:
      return Expr(SymEngine::make_rcp<const SymEngine::LessThan>(l, r));
    case Rel::Gt:
      return Expr(SymEngine::make_rcp<const SymEngine::StrictLessThan>(r, l));
    case Rel::Ge:
      return Expr(SymEngine::make_rcp<const SymEngine::LessThan>(r, l));
  }
  TORCH_INTERNAL_ASSERT(false);
}

Expr logical_not(const Expr& a) {
  return Expr(SymEngine::logical_not(as_boolean(a)));
}

Expr logical_and(c10::ArrayRef<Expr> args) {
  return Expr(SymEngine::logical_and(to_set(args)));
}

Expr logical_or(c10::ArrayRef<Expr> args) {
  return Expr(SymEngine::logical_or(to_set(args)));
}

Expr SymbolTable::symbol(const std::string& name, const SymbolFacts& facts) {
  auto [it, inserted] = facts_.try_emplace(name, facts);
  const SymbolFacts& f = it->second;
  if (!inserted &&
      (f.integer != facts.integer || f.nonnegative != facts.nonnegative ||
       f.positive != facts.positive)) {
    throw NativeUnsupported("symbol " + name + " with different assumptions");
  }
  return Expr(SymEngine::symbol(name));
}

const SymbolFacts* SymbolTable::facts(const Expr& sym) const {
  auto it = facts_.find(symbol_name(sym));
  return it == facts_.end() ? nullptr : &it->second;
}

bool SymbolTable::nonnegative(const Expr& e) const {
  const Basic& b = *e.get_basic();
  if (is_number(e)) {
    return compare_numbers(e, Expr(0)) >= 0;
  }
  if (is_symbol(e)) {
    const SymbolFacts* f = facts(e);
    return f != nullptr && (f->nonnegative || f->positive);
  }
  auto all_args = [&](auto pred) {
    auto a = args(e);
    return std::all_of(a.begin(), a.end(), pred);
  };
  auto nonneg = [&](const Expr& x) { return nonnegative(x); };
  if (is_a<SymEngine::Add>(b) || is_a<SymEngine::Mul>(b) ||
      is_a<SymEngine::Min>(b)) {
    return all_args(nonneg);
  }
  if (is_a<SymEngine::Max>(b)) {
    auto a = args(e);
    return std::any_of(a.begin(), a.end(), nonneg);
  }
  if (is_a<SymEngine::Pow>(b)) {
    auto a = args(e);
    return nonnegative(a[0]);
  }
  if (const TorchFunction* f = as_function(e)) {
    auto a = args(e);
    switch (f->fn()) {
      case Fn::FloorDiv:
      case Fn::CleanDiv:
        return nonnegative(a[0]) && positive(a[1]);
      case Fn::Mod:
      case Fn::IsNonOverlappingAndDenseIndicator:
        return true;
      case Fn::PythonMod:
        return positive(a[1]);
      case Fn::PowByNatural:
        return nonnegative(a[0]);
    }
  }
  return false;
}

bool SymbolTable::positive(const Expr& e) const {
  const Basic& b = *e.get_basic();
  if (is_number(e)) {
    return compare_numbers(e, Expr(0)) > 0;
  }
  if (is_symbol(e)) {
    const SymbolFacts* f = facts(e);
    return f != nullptr && f->positive;
  }
  auto a = args(e);
  auto pos = [&](const Expr& x) { return positive(x); };
  auto nonneg = [&](const Expr& x) { return nonnegative(x); };
  if (is_a<SymEngine::Add>(b)) {
    return std::all_of(a.begin(), a.end(), nonneg) &&
        std::any_of(a.begin(), a.end(), pos);
  }
  if (is_a<SymEngine::Mul>(b) || is_a<SymEngine::Min>(b)) {
    return std::all_of(a.begin(), a.end(), pos);
  }
  if (is_a<SymEngine::Max>(b)) {
    return std::any_of(a.begin(), a.end(), pos);
  }
  if (is_a<SymEngine::Pow>(b) || is_function(e, Fn::PowByNatural)) {
    return positive(a[0]);
  }
  return false;
}

} // namespace torch::symbolic
