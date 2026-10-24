#pragma once

#include <torch/csrc/symbolic/ValueRanges.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

// The part of a Python ShapeEnv that native evaluation reads, mirrored by the
// Python side (Python stays authoritative), and native versions of the
// ShapeEnv evaluation steps. Static evaluation answers only while the env is
// pristine: no guards, deferred runtime asserts, axioms, divisible,
// replacements or post-creation range updates.

namespace torch::symbolic {

// None, int or bool, as passed to ShapeEnv.evaluate_expr.
using Hint = std::variant<std::monostate, int64_t, bool>;

// An expression or None.
using MaybeExpr = std::optional<Expr>;

// A query answered natively. Python replays it (_flush_native_queries) before
// it next evaluates or mutates any ShapeEnv, so that its caches, which do not
// key on guards, axioms or ranges and are shared by all envs, hold the
// entries a run without native answers would.
struct NativeQuery {
  Expr expr;
  // _maybe_evaluate_static(expr) if false, else evaluate_expr(expr, hint,
  // fx_node=False, size_oblivious=False, fallback_value) under the
  // suppress_guards TLS.
  bool evaluate;
  Hint hint;
  std::optional<bool> fallback_value;
  bool suppress_guards;

  bool operator==(const NativeQuery& o) const {
    return expr == o.expr && evaluate == o.evaluate && hint == o.hint &&
        fallback_value == o.fallback_value &&
        suppress_guards == o.suppress_guards;
  }
};

struct LoggedQuery {
  // Position in the last-use order over all envs.
  uint64_t seq = 0;
  NativeQuery query;
  MaybeExpr result;
};

// Mirror of the Python ShapeEnv suppress_guards TLS.
bool& suppress_guards_tls();

class NativeShapeEnv : public c10::intrusive_ptr_target {
 public:
  explicit NativeShapeEnv(c10::intrusive_ptr<SymbolTable> symbols)
      : symbols_(std::move(symbols)) {}
  ~NativeShapeEnv() override;

  SymbolTable& symbols() {
    return *symbols_;
  }
  // Guards the env and its symbol table (PyFallback.h: lock_env).
  std::mutex& mutex() {
    return mutex_;
  }
  // Python-side state (python_symbolic.cpp), opaque to the core.
  void set_binding(std::shared_ptr<void> binding) {
    binding_ = std::move(binding);
  }
  void* binding() const {
    return binding_.get();
  }
  // Native nodes of this env that are alive. While there are any, the binding
  // holds the Python ShapeEnv strongly, as a Python SymNode does.
  std::atomic<int64_t>& live_nodes() {
    return live_nodes_;
  }

  void add_symbol(
      const Expr& sym,
      std::optional<int64_t> hint,
      const ValueRanges& range);
  void update_range(const Expr& sym, const ValueRanges& range);
  void mark_not_pristine() {
    pristine_ = false;
  }
  void mark_replacements() {
    replacements_empty_ = false;
    pristine_ = false;
  }
  bool pristine() const {
    return pristine_;
  }
  bool replacements_empty() const {
    return replacements_empty_;
  }
  // Native mod results by (lhs, rhs), recorded while pristine. The
  // Mod/PythonMod choice reads ranges and _symop_cache keeps the first
  // answer, so the binding writes these into _symop_cache before the env
  // leaves pristine, and afterwards a range-dependent choice is native only
  // when memoized here.
  using ModMemo = std::unordered_map<std::pair<Expr, Expr>, Expr, ExprPairHash>;
  ModMemo& mod_memo() {
    return mod_memo_;
  }
  // (hint, range) of a mirrored symbol.
  std::optional<std::pair<std::optional<int64_t>, ValueRanges>> mirrored(
      const Expr& sym) const;

  // ShapeEnv.simplify(e) and ShapeEnv._maybe_evaluate_static(e), which
  // require a pristine env with every free symbol mirrored, and
  // ShapeEnv._maybe_fast_eval_comparison(e).
  Expr simplify(const Expr& e);
  MaybeExpr maybe_evaluate_static(const Expr& e);
  MaybeExpr maybe_fast_eval_comparison(const Expr& e);
  // ShapeEnv.bound_sympy(e).lower >= 0. Throws NativeUnsupported unless the
  // env is pristine and every free symbol is mirrored.
  bool bound_lower_nonnegative(const Expr& e);

  // Entry points; nullopt means the caller must delegate to Python.
  // Answers are logged for replay.
  // _maybe_evaluate_static(e) as _static_eval_sym_bool calls it.
  std::optional<MaybeExpr> static_eval(const Expr& e);
  // ShapeEnv.evaluate_expr(e, hint, fallback_value=fallback_value) where it
  // returns without adding a guard. The caller checks that
  // aggressive_guard_free_semantics and backed_size_oblivious are unset.
  std::optional<MaybeExpr> evaluate_expr(
      const Expr& e,
      const Hint& hint,
      std::optional<bool> fallback_value = std::nullopt);

  // Distinct answered queries since the last call, in last-use order.
  std::vector<LoggedQuery> take_queries();
  // Whether any env has queries to replay.
  static bool queries_pending();

 private:
  struct QueryHash {
    size_t operator()(const NativeQuery& q) const;
  };

  std::optional<MaybeExpr> evaluate_expr_impl(const Expr& e, const Hint& hint);
  MaybeExpr maybe_evaluate_static_uncached(const Expr& e);
  void log_query(const NativeQuery& q, const MaybeExpr& result);
  bool all_symbols_mirrored(const Expr& e) const;
  // Whether a <= b follows from the ranges.
  bool definitely_le(const Expr& a, const Expr& b);

  c10::intrusive_ptr<SymbolTable> symbols_;
  std::mutex mutex_;
  std::shared_ptr<void> binding_;
  std::atomic<int64_t> live_nodes_{0};
  RangeMap var_to_range_;
  ExprMap<int64_t> backed_var_to_val_;
  bool pristine_ = true;
  bool replacements_empty_ = true;
  ModMemo mod_memo_;
  ExprMap<MaybeExpr> static_memo_;
  std::unordered_map<NativeQuery, std::pair<uint64_t, MaybeExpr>, QueryHash>
      queries_;
};

} // namespace torch::symbolic
