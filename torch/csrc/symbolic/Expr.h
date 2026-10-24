#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>

#include <symengine/expression.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Symbolic expressions for torch's symbolic shapes, backed by SymEngine.
// Expressions need not be structurally identical to what sympy builds: they
// are converted to sympy (python_symbolic.cpp) whenever Python needs them.
// Anything that is not supported throws NativeUnsupported and the caller
// falls back to Python.

namespace torch::symbolic {

using Expr = SymEngine::Expression;

struct NativeUnsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ExprHash {
  size_t operator()(const Expr& e) const {
    return e.get_basic()->hash();
  }
};

template <typename V>
using ExprMap = std::unordered_map<Expr, V, ExprHash>;

struct ExprPairHash {
  size_t operator()(const std::pair<Expr, Expr>& p) const;
};

// int_oo and -int_oo, as SymEngine's oo and -oo. sympy's float oo is never
// represented.
Expr int_oo();
Expr neg_int_oo();
Expr boolean(bool v);

bool is_int_oo(const Expr& e);
bool is_integer(const Expr& e);
bool is_symbol(const Expr& e);
// An Integer, Rational or +-int_oo.
bool is_number(const Expr& e);
// BooleanAtom, Relational or a logical connective.
bool is_boolean(const Expr& e);
bool is_relational(const Expr& e);
std::optional<bool> as_bool(const Expr& e);
std::optional<int64_t> as_int64(const Expr& e);
const std::string& symbol_name(const Expr& e);

std::vector<Expr> args(const Expr& e);
std::vector<Expr> free_symbols(const Expr& e);
using SymEngine::expand;
Expr xreplace(const Expr& e, c10::ArrayRef<std::pair<Expr, Expr>> reps);
Expr add(c10::ArrayRef<Expr> args);
Expr mul(c10::ArrayRef<Expr> args);

// e as coeff * term with a number coeff.
std::pair<Expr, Expr> as_coeff_mul(const Expr& e);

// Three-way comparison of numbers, int_oo-aware.
int compare_numbers(const Expr& a, const Expr& b);

enum class Rel : uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

// Rel(lhs, rhs). With `evaluate`, decides it when rhs - lhs is a number.
Expr rel(Rel kind, const Expr& lhs, const Expr& rhs, bool evaluate = true);
Expr logical_not(const Expr& a);
Expr logical_and(c10::ArrayRef<Expr> args);
Expr logical_or(c10::ArrayRef<Expr> args);

// Assumptions of a symbol, as sympy's assumptions0 sets them.
struct SymbolFacts {
  bool integer = false;
  bool nonnegative = false;
  bool positive = false;
};

// The facts of the symbols of an env's expressions. SymEngine symbols carry
// only a name.
class SymbolTable : public c10::intrusive_ptr_target {
 public:
  // The symbol named `name`; throws if it exists with different facts.
  Expr symbol(const std::string& name, const SymbolFacts& facts);
  const SymbolFacts* facts(const Expr& sym) const;
  // Whether e >= 0 follows from the facts of its symbols.
  bool nonnegative(const Expr& e) const;
  bool positive(const Expr& e) const;

 private:
  std::unordered_map<std::string, SymbolFacts> facts_;
};

} // namespace torch::symbolic
