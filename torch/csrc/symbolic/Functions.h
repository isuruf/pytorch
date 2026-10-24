#pragma once

#include <torch/csrc/symbolic/Expr.h>

#include <symengine/functions.h>

// The classes of torch/utils/_sympy/functions.py that native nodes build, as
// SymEngine FunctionWrappers named after their sympy class. Max and Min are
// SymEngine's.

namespace torch::symbolic {

enum class Fn : uint8_t {
  FloorDiv,
  CleanDiv,
  Mod,
  PythonMod,
  PowByNatural,
  IsNonOverlappingAndDenseIndicator,
};

const char* function_name(Fn fn);
std::optional<Fn> function_from_name(const std::string& name);

class TorchFunction : public SymEngine::FunctionWrapper {
 public:
  TorchFunction(Fn fn, const SymEngine::vec_basic& args);

  Fn fn() const {
    return fn_;
  }

  // Rebuilds through function(), so substitution re-evaluates.
  SymEngine::RCP<const SymEngine::Basic> create(
      const SymEngine::vec_basic& args) const override;
  SymEngine::RCP<const SymEngine::Number> eval(long bits) const override;
  SymEngine::RCP<const SymEngine::Basic> diff_impl(
      const SymEngine::RCP<const SymEngine::Symbol>& s) const override;
  // FunctionSymbol's only match FunctionSymbol, not its subclasses.
  bool __eq__(const SymEngine::Basic& o) const override;
  int compare(const SymEngine::Basic& o) const override;

 private:
  Fn fn_;
};

// The TorchFunction e is, or nullptr.
const TorchFunction* as_function(const Expr& e);
bool is_function(const Expr& e, Fn fn);

// cls(*args) for the class of fn: its eval's result when there is one, else
// the unevaluated function. Throws NativeUnsupported where Python would raise
// (a zero divisor).
Expr function(Fn fn, c10::ArrayRef<Expr> args);
Expr sym_max(c10::ArrayRef<Expr> args);
Expr sym_min(c10::ArrayRef<Expr> args);
bool is_max(const Expr& e);
bool is_min(const Expr& e);

// symbolic_shapes.eval_is_non_overlapping_and_dense.
int64_t eval_is_non_overlapping_and_dense(
    c10::ArrayRef<int64_t> sizes,
    c10::ArrayRef<int64_t> strides);

} // namespace torch::symbolic
