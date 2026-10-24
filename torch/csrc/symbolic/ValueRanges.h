#pragma once

#include <torch/csrc/symbolic/Expr.h>

// Value range analysis in the spirit of torch/utils/_sympy/value_ranges.py,
// for integer and boolean expressions.

namespace torch::symbolic {

// Integer bounds are Integers or +-int_oo; boolean bounds are true/false.
struct ValueRanges {
  // Throws for an invalid range.
  ValueRanges(Expr lower, Expr upper);

  static ValueRanges boolean_range(bool can_be_false, bool can_be_true);

  bool is_bool() const;
  bool is_int() const;
  bool is_singleton() const {
    return lower == upper;
  }
  // For a boolean range.
  bool can_be_true() const {
    return upper == boolean(true);
  }
  bool can_be_false() const {
    return lower == boolean(false);
  }

  Expr lower;
  Expr upper;
};

using RangeMap = ExprMap<ValueRanges>;

// The range of e given the ranges of its symbols. A symbol without a range
// gets the range its facts imply. Throws NativeUnsupported for expressions
// the analysis does not handle.
ValueRanges bound_sympy(
    const Expr& e,
    const RangeMap& ranges,
    const SymbolTable& symbols);

} // namespace torch::symbolic
