#include <torch/csrc/symbolic/NativeShapeEnv.h>

#include <torch/csrc/symbolic/Functions.h>

#include <c10/util/hash.h>

#include <symengine/add.h>
#include <symengine/logic.h>

#include <algorithm>
#include <atomic>

namespace torch::symbolic {

namespace {

// The Max and Min atoms of e.
std::vector<Expr> min_max_atoms(const Expr& e) {
  std::vector<Expr> out;
  std::vector<Expr> stack{e};
  while (!stack.empty()) {
    Expr t = stack.back();
    stack.pop_back();
    if ((is_max(t) || is_min(t)) &&
        std::find(out.begin(), out.end(), t) == out.end()) {
      out.push_back(t);
    }
    for (Expr& a : args(t)) {
      stack.push_back(std::move(a));
    }
  }
  return out;
}

std::atomic<uint64_t> query_seq{0};
std::atomic<int64_t> pending_queries{0};

} // namespace

bool& suppress_guards_tls() {
  thread_local bool value = false;
  return value;
}

NativeShapeEnv::~NativeShapeEnv() {
  pending_queries -= static_cast<int64_t>(queries_.size());
}

size_t NativeShapeEnv::QueryHash::operator()(const NativeQuery& q) const {
  return c10::get_hash(
      ExprHash()(q.expr),
      q.evaluate,
      q.hint.index(),
      std::visit(
          [](auto v) -> int64_t {
            if constexpr (std::is_same_v<decltype(v), std::monostate>) {
              return 0;
            } else {
              return static_cast<int64_t>(v);
            }
          },
          q.hint),
      q.fallback_value.has_value(),
      q.fallback_value.value_or(false),
      q.suppress_guards);
}

void NativeShapeEnv::log_query(const NativeQuery& q, const MaybeExpr& result) {
  auto [it, inserted] = queries_.try_emplace(q);
  if (inserted) {
    ++pending_queries;
  }
  it->second = {++query_seq, result};
}

std::vector<LoggedQuery> NativeShapeEnv::take_queries() {
  std::vector<LoggedQuery> out;
  out.reserve(queries_.size());
  for (const auto& [q, v] : queries_) {
    out.push_back({v.first, q, v.second});
  }
  pending_queries -= static_cast<int64_t>(queries_.size());
  queries_.clear();
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.seq < b.seq;
  });
  return out;
}

bool NativeShapeEnv::queries_pending() {
  return pending_queries > 0;
}

void NativeShapeEnv::add_symbol(
    const Expr& sym,
    std::optional<int64_t> hint,
    const ValueRanges& range) {
  if (!is_symbol(sym) || !range.is_int()) {
    throw NativeUnsupported("add_symbol needs an integer symbol");
  }
  TORCH_CHECK(var_to_range_.count(sym) == 0, "symbol already mirrored");
  var_to_range_.insert_or_assign(sym, range);
  if (hint) {
    backed_var_to_val_[sym] = *hint;
  }
}

void NativeShapeEnv::update_range(const Expr& sym, const ValueRanges& range) {
  pristine_ = false;
  if (!range.is_int()) {
    var_to_range_.erase(sym);
    throw NativeUnsupported("update_range needs an integer range");
  }
  var_to_range_.insert_or_assign(sym, range);
}

std::optional<std::pair<std::optional<int64_t>, ValueRanges>> NativeShapeEnv::
    mirrored(const Expr& sym) const {
  auto it = var_to_range_.find(sym);
  if (it == var_to_range_.end()) {
    return std::nullopt;
  }
  auto hint = backed_var_to_val_.find(sym);
  return std::make_pair(
      hint == backed_var_to_val_.end() ? std::nullopt
                                       : std::optional<int64_t>(hint->second),
      it->second);
}

bool NativeShapeEnv::all_symbols_mirrored(const Expr& e) const {
  auto fs = free_symbols(e);
  return std::all_of(fs.begin(), fs.end(), [&](const Expr& s) {
    return var_to_range_.count(s) != 0;
  });
}

bool NativeShapeEnv::definitely_le(const Expr& a, const Expr& b) {
  if (a == b) {
    return true;
  }
  try {
    ValueRanges d = bound_sympy(expand(b - a), var_to_range_, *symbols_);
    return compare_numbers(d.lower, Expr(0)) >= 0;
  } catch (const NativeUnsupported&) {
    return false;
  }
}

Expr NativeShapeEnv::simplify(const Expr& e_) {
  if (!pristine_ || !all_symbols_mirrored(e_)) {
    throw NativeUnsupported("simplify needs a pristine, mirrored env");
  }
  // replace() is the identity while there are no replacements.
  Expr e = expand(e_);
  // Drop the Max/Min args that another arg dominates.
  std::vector<std::pair<Expr, Expr>> reps;
  for (const Expr& atom : min_max_atoms(e)) {
    bool max = is_max(atom);
    auto a = args(atom);
    std::vector<bool> keep(a.size(), true);
    for (size_t i = 0; i < a.size(); ++i) {
      for (size_t j = 0; j < a.size() && keep[i]; ++j) {
        if (i == j || !keep[j]) {
          continue;
        }
        // a[i] is redundant when a[j] dominates it.
        if (max ? definitely_le(a[i], a[j]) : definitely_le(a[j], a[i])) {
          keep[i] = false;
        }
      }
    }
    std::vector<Expr> kept;
    for (size_t i = 0; i < a.size(); ++i) {
      if (keep[i]) {
        kept.push_back(a[i]);
      }
    }
    if (kept.size() != a.size()) {
      reps.emplace_back(atom, max ? sym_max(kept) : sym_min(kept));
    }
  }
  return reps.empty() ? e : xreplace(e, reps);
}

MaybeExpr NativeShapeEnv::maybe_evaluate_static(const Expr& e) {
  // Answers need a pristine env (simplify throws otherwise), whose mirrored
  // ranges never change, so they depend only on e.
  if (pristine_) {
    if (auto it = static_memo_.find(e); it != static_memo_.end()) {
      return it->second;
    }
  }
  MaybeExpr r = maybe_evaluate_static_uncached(e);
  static_memo_.emplace(e, r);
  return r;
}

MaybeExpr NativeShapeEnv::maybe_evaluate_static_uncached(const Expr& e_) {
  Expr e = simplify(e_);
  if (is_number(e) || as_bool(e)) {
    return e;
  }
  ValueRanges out = bound_sympy(e, var_to_range_, *symbols_);
  if (!out.is_singleton()) {
    return std::nullopt;
  }
  return out.lower;
}

MaybeExpr NativeShapeEnv::maybe_fast_eval_comparison(const Expr& e) {
  // Ge(sum, 0) and Le(0, sum) are both LessThan(0, sum).
  if (!SymEngine::is_a<SymEngine::LessThan>(*e.get_basic())) {
    return std::nullopt;
  }
  auto a = args(e);
  if (a[0] != Expr(0)) {
    return std::nullopt;
  }
  // _is_nonneg_sum.
  auto nonneg_term = [&](const Expr& term) {
    if (is_symbol(term)) {
      auto it = var_to_range_.find(term);
      return it != var_to_range_.end() &&
          compare_numbers(it->second.lower, Expr(0)) >= 0;
    }
    return is_number(term) && compare_numbers(term, Expr(0)) >= 0;
  };
  auto terms = SymEngine::is_a<SymEngine::Add>(*a[1].get_basic())
      ? args(a[1])
      : std::vector<Expr>{a[1]};
  if (std::all_of(terms.begin(), terms.end(), nonneg_term)) {
    return boolean(true);
  }
  return std::nullopt;
}

bool NativeShapeEnv::bound_lower_nonnegative(const Expr& e) {
  if (!pristine_ || !all_symbols_mirrored(e)) {
    throw NativeUnsupported("bound_sympy needs a pristine, mirrored env");
  }
  const Expr lower = bound_sympy(e, var_to_range_, *symbols_).lower;
  return compare_numbers(lower, Expr(0)) >= 0;
}

std::optional<MaybeExpr> NativeShapeEnv::static_eval(const Expr& e) {
  if (!pristine_ || !all_symbols_mirrored(e)) {
    return std::nullopt;
  }
  MaybeExpr r;
  try {
    r = maybe_evaluate_static(e);
  } catch (const NativeUnsupported&) {
    return std::nullopt;
  }
  log_query({e, false, std::monostate{}, std::nullopt, false}, r);
  return r;
}

std::optional<MaybeExpr> NativeShapeEnv::evaluate_expr(
    const Expr& e,
    const Hint& hint,
    std::optional<bool> fallback_value) {
  auto r = evaluate_expr_impl(e, hint);
  if (r) {
    log_query({e, true, hint, fallback_value, suppress_guards_tls()}, *r);
  }
  return r;
}

std::optional<MaybeExpr> NativeShapeEnv::evaluate_expr_impl(
    const Expr& e,
    const Hint& hint) {
  // _evaluate_expr up to the static evaluation; the rest adds a guard or
  // raises.
  if (as_bool(e)) {
    return MaybeExpr(e);
  }
  // Python evaluates the replaced expr.
  if (!replacements_empty_ || !all_symbols_mirrored(e)) {
    return std::nullopt;
  }
  if (is_number(e)) {
    // Python asserts that e equals the hint (Integer(1) == True).
    if (std::holds_alternative<std::monostate>(hint)) {
      return MaybeExpr(e);
    }
    const auto* i = std::get_if<int64_t>(&hint);
    if (as_int64(e) == (i ? *i : int64_t(std::get<bool>(hint)))) {
      return MaybeExpr(e);
    }
    return std::nullopt;
  }
  // The range mirror is not updated once the env stops being pristine.
  if (!pristine_) {
    return std::nullopt;
  }
  try {
    if (MaybeExpr r = maybe_fast_eval_comparison(e)) {
      return r;
    }
    // Without a static answer Python guards on the hint.
    if (MaybeExpr r = maybe_evaluate_static(e)) {
      return r;
    }
    return std::nullopt;
  } catch (const NativeUnsupported&) {
    return std::nullopt;
  }
}

} // namespace torch::symbolic
