#include <torch/csrc/symbolic/NativeSymNodeImpl.h>

#include <ATen/PythonTorchFunctionTLS.h>

#include <torch/csrc/symbolic/Functions.h>
#include <torch/csrc/symbolic/PyFallback.h>

#include <algorithm>
#include <array>
#include <limits>

namespace torch::symbolic {

namespace {

NativeSymNodeImpl* as_native(const c10::SymNode& node) {
  return dynamic_cast<NativeSymNodeImpl*>(node.get());
}

c10::SymNode to_python(const c10::SymNode& node) {
  auto* native = as_native(node);
  return native ? materialize(*native) : node;
}

std::vector<c10::SymNode> to_python(c10::ArrayRef<c10::SymNode> nodes) {
  std::vector<c10::SymNode> out;
  out.reserve(nodes.size());
  for (const auto& n : nodes) {
    out.push_back(to_python(n));
  }
  return out;
}

// Python int power; nullopt on a negative exponent (a float) or overflow.
std::optional<int64_t> int_pow(int64_t base, int64_t exp) {
  if (exp < 0) {
    return std::nullopt;
  }
  int64_t r = 1;
  while (exp != 0) {
    if ((exp & 1) && __builtin_mul_overflow(r, base, &r)) {
      return std::nullopt;
    }
    exp >>= 1;
    if (exp != 0 && __builtin_mul_overflow(base, base, &base)) {
      return std::nullopt;
    }
  }
  return r;
}

// bool() of an evaluate_expr result; nullopt for no result.
std::optional<bool> truth(const MaybeExpr& e) {
  if (!e) {
    return std::nullopt;
  }
  if (auto b = as_bool(*e)) {
    return b;
  }
  if (auto i = as_int64(*e)) {
    return *i != 0;
  }
  return std::nullopt;
}

// sym_node.sympy_is_contiguous_generic.
Expr contiguous_generic(
    c10::ArrayRef<Expr> sizes,
    c10::ArrayRef<Expr> strides,
    c10::ArrayRef<size_t> dim_order) {
  if (dim_order.size() != sizes.size()) {
    return boolean(false);
  }
  Expr one(1);
  Expr r = boolean(true);
  Expr z = one;
  for (size_t d : dim_order) {
    r = logical_and(
        {r,
         logical_or(
             {rel(Rel::Eq, sizes[d], one), rel(Rel::Eq, strides[d], z)})});
    z = z * sizes[d];
  }
  for (const Expr& s : sizes) {
    r = logical_or({r, rel(Rel::Eq, s, Expr(0))});
  }
  return r;
}

// sym_node.sympy_is_channels_last_strides_generic.
Expr channels_last_strides_generic(
    c10::ArrayRef<Expr> sizes,
    c10::ArrayRef<Expr> strides,
    c10::ArrayRef<size_t> dim_order) {
  if (dim_order.size() != sizes.size()) {
    return boolean(false);
  }
  Expr zero(0);
  Expr m = zero;
  Expr r = rel(Rel::Ne, strides[1], zero);
  for (size_t d : dim_order) {
    r = logical_and(
        {r, rel(Rel::Ne, sizes[d], zero), rel(Rel::Ge, strides[d], m)});
    if (d == 0) {
      r = logical_and({r, rel(Rel::Ne, m, strides[1])});
    }
    m = strides[d] * sym_max({sizes[d], Expr(1)});
  }
  return r;
}

// e if it is not decided by the facts of its symbols, as sympy evaluates
// relationals with the assumptions of the symbols, else its value.
Expr fold(const Expr& e, const SymbolTable& symbols) {
  try {
    ValueRanges r = bound_sympy(e, {}, symbols);
    return r.is_singleton() ? r.lower : e;
  } catch (const NativeUnsupported&) {
    return e;
  }
}

// f(), or null where it needs what the native side lacks.
template <typename F>
c10::SymNode native_or_null(F f) {
  try {
    return f();
  } catch (const NativeUnsupported&) {
    return {};
  }
}

// sym_min/sym_max/sym_not reach a TorchFunctionMode through has_torch_function.
bool torch_function_pending() {
  return at::impl::torch_function_mode_enabled() ||
      at::impl::PythonTorchFunctionTLS::peek_skip_next();
}

} // namespace

NativeSymNodeImpl::NativeSymNodeImpl(
    c10::intrusive_ptr<NativeShapeEnv> env,
    Expr expr,
    PyType pytype,
    Hint hint,
    Hint constant)
    : env_(std::move(env)),
      expr_(std::move(expr)),
      pytype_(pytype),
      hint_(hint),
      constant_(constant) {
  if (env_->live_nodes().fetch_add(1) == 0) {
    live_nodes_changed(*env_);
  }
}

NativeSymNodeImpl::~NativeSymNodeImpl() {
  if (env_->live_nodes().fetch_sub(1) == 1) {
    live_nodes_changed(*env_);
  }
}

std::optional<int64_t> NativeSymNodeImpl::maybe_as_int() {
  {
    auto lock = lock_env(*env_);
    if (env_->replacements_empty()) {
      if (auto i = as_int64(expr_)) {
        return i;
      }
      if (!is_number(expr_)) {
        return std::nullopt;
      }
    }
  }
  return materialize(*this)->maybe_as_int();
}

std::string NativeSymNodeImpl::str() {
  bool replacements_empty = false;
  {
    auto lock = lock_env(*env_);
    replacements_empty = env_->replacements_empty();
  }
  if (replacements_empty) {
    return sympy_str(*this);
  }
  return materialize(*this)->str();
}

c10::SymNode NativeSymNodeImpl::wrap_int(int64_t num) {
  auto lock = lock_env(*env_);
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, Expr(num), PyType::Int, num, num);
}

c10::SymNode NativeSymNodeImpl::wrap_float(double num) {
  return materialize(*this)->wrap_float(num);
}

c10::SymNode NativeSymNodeImpl::wrap_bool(bool num) {
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, boolean(num), PyType::Bool, num, num);
}

c10::SymNode NativeSymNodeImpl::binary(
    Op op,
    BinaryFn fallback,
    const c10::SymNode& other) {
  static constexpr std::array proxy_methods = {
      "_add",
      "_sub",
      "_mul",
      "_int_floordiv",
      "_mod",
      "_pow_by_natural",
      "_sym_min",
      "_sym_max",
      "_eq",
      "_ne",
      "_gt",
      "_lt",
      "_le",
      "_ge",
      "_and_",
      "_or_"};
  static constexpr std::array op_names = {
      "add",
      "sub",
      "mul",
      "int_floordiv",
      "mod",
      "pow_by_natural",
      "sym_min",
      "sym_max",
      "eq",
      "ne",
      "gt",
      "lt",
      "le",
      "ge",
      "and",
      "or"};
  static_assert(std::size(proxy_methods) == static_cast<size_t>(Op::Or) + 1);
  static_assert(std::size(op_names) == std::size(proxy_methods));
  auto* o = as_native(other);
  if (proxy_mode()) {
    auto i = static_cast<size_t>(op);
    // The Python impl first computes op(self.hint, other.hint), which raises
    // on a zero divisor and may not terminate for pow.
    bool python_hint = op == Op::PowByNatural ||
        ((op == Op::FloorDiv || op == Op::Mod) && has_hint() && o != nullptr &&
         o->has_hint() &&
         std::visit(
             [](auto v) {
               if constexpr (std::is_same_v<decltype(v), std::monostate>) {
                 return false;
               } else {
                 return v == 0;
               }
             },
             o->hint()));
    if (o == nullptr || python_hint) {
      return python_impl(proxy_methods[i], {clone(), other});
    }
    return proxy_dispatch(
        proxy_methods[i], op_names[i], {clone(), other}, op == Op::Mul, [&] {
          return proxy_compute(op, *o);
        });
  }
  if (o != nullptr && o->env_ == env_) {
    auto lock = lock_env(*env_);
    if (auto r = native_or_null([&] { return try_binary(op, *o); })) {
      return r;
    }
  }
  return (materialize(*this).get()->*fallback)(to_python(other));
}

// Python evaluates op(lhs, rhs) with a constant lhs through the method of rhs:
// the reflected one computes lhs.op(rhs), a comparison or sym_min/sym_max
// computes the mirrored rhs.op(lhs). Two constants are Python arithmetic.
c10::SymNode NativeSymNodeImpl::proxy_compute(Op op, NativeSymNodeImpl& other) {
  bool constant_lhs = !std::holds_alternative<std::monostate>(constant_);
  if (other.env_ != env_ ||
      (constant_lhs &&
       !std::holds_alternative<std::monostate>(other.constant_)) ||
      ((op == Op::Min || op == Op::Max) && torch_function_pending())) {
    return {};
  }
  NativeSymNodeImpl* lhs = this;
  NativeSymNodeImpl* rhs = &other;
  if (constant_lhs) {
    switch (op) {
      case Op::Gt:
        op = Op::Lt;
        std::swap(lhs, rhs);
        break;
      case Op::Lt:
        op = Op::Gt;
        std::swap(lhs, rhs);
        break;
      case Op::Le:
        op = Op::Ge;
        std::swap(lhs, rhs);
        break;
      case Op::Ge:
        op = Op::Le;
        std::swap(lhs, rhs);
        break;
      case Op::Eq:
      case Op::Ne:
      case Op::Min:
      case Op::Max:
        std::swap(lhs, rhs);
        break;
      default:
        break;
    }
  }
  auto lock = lock_env(*env_);
  return native_or_null([&] { return lhs->try_binary(op, *rhs); });
}

c10::SymNode NativeSymNodeImpl::try_binary(
    Op op,
    const NativeSymNodeImpl& other) {
  bool logical = op == Op::And || op == Op::Or;
  PyType operand = logical ? PyType::Bool : PyType::Int;
  if (!env_->replacements_empty() || pytype_ != operand ||
      other.pytype_ != operand) {
    return {};
  }

  // The hint first, as Python computes it; a hint that Python cannot
  // represent natively or that raises goes to the fallback.
  Hint hint;
  if (has_hint() && !std::holds_alternative<std::monostate>(other.hint_)) {
    if (logical) {
      bool x = std::get<bool>(hint_);
      bool y = std::get<bool>(other.hint_);
      hint = op == Op::And ? (x && y) : (x || y);
    } else {
      int64_t x = std::get<int64_t>(hint_);
      int64_t y = std::get<int64_t>(other.hint_);
      int64_t r = 0;
      switch (op) {
        case Op::Add:
          if (__builtin_add_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::Sub:
          if (__builtin_sub_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::Mul:
          if (__builtin_mul_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::FloorDiv:
          if (y == 0 || (x == std::numeric_limits<int64_t>::min() && y == -1)) {
            return {};
          }
          r = x / y;
          if (x % y != 0 && ((x < 0) != (y < 0))) {
            --r;
          }
          hint = r;
          break;
        case Op::Mod:
          if (y == 0) {
            return {};
          }
          r = y == -1 ? 0 : x % y;
          if (r != 0 && ((r < 0) != (y < 0))) {
            r += y;
          }
          hint = r;
          break;
        case Op::PowByNatural: {
          auto p = int_pow(x, y);
          if (!p) {
            return {};
          }
          hint = *p;
          break;
        }
        case Op::Min:
          hint = std::min(x, y);
          break;
        case Op::Max:
          hint = std::max(x, y);
          break;
        case Op::Eq:
          hint = x == y;
          break;
        case Op::Ne:
          hint = x != y;
          break;
        case Op::Gt:
          hint = x > y;
          break;
        case Op::Lt:
          hint = x < y;
          break;
        case Op::Le:
          hint = x <= y;
          break;
        case Op::Ge:
          hint = x >= y;
          break;
        case Op::And:
        case Op::Or:
          break;
      }
    }
  }

  const Expr& a = expr_;
  const Expr& b = other.expr_;
  // IntInfinity overloads the Python operators.
  if ((op == Op::Add || op == Op::Sub || op == Op::Mul) &&
      (is_int_oo(a) || is_int_oo(b))) {
    return {};
  }
  Expr out;
  switch (op) {
    case Op::Add:
      out = a + b;
      break;
    case Op::Sub:
      out = a - b;
      break;
    case Op::Mul:
      out = a * b;
      break;
    case Op::FloorDiv:
      out = function(Fn::FloorDiv, {a, b});
      break;
    case Op::Mod: {
      // bound_lower_nonnegative throws once the env is not pristine, so then
      // only the memo and the range-free answer are native.
      auto& memo = env_->mod_memo();
      auto it = memo.find({a, b});
      if (it != memo.end()) {
        out = it->second;
        break;
      }
      auto nonnegative = [&](const Expr& e) {
        return env_->symbols().nonnegative(e) ||
            env_->bound_lower_nonnegative(e);
      };
      bool mod = nonnegative(a) && nonnegative(b);
      out = function(mod ? Fn::Mod : Fn::PythonMod, {a, b});
      if (env_->pristine()) {
        memo.emplace(std::make_pair(a, b), out);
      }
      break;
    }
    case Op::PowByNatural:
      out = function(Fn::PowByNatural, {a, b});
      break;
    case Op::Min:
    case Op::Max: {
      // As sympy's Max and Min use the assumptions of the symbols.
      auto le = as_bool(fold(rel(Rel::Le, a, b), env_->symbols()));
      if (!le && as_bool(fold(rel(Rel::Le, b, a), env_->symbols())) == true) {
        le = false;
      }
      if (le) {
        out = (op == Op::Min) == *le ? a : b;
      } else {
        out = op == Op::Min ? symbolic::sym_min({a, b})
                            : symbolic::sym_max({a, b});
      }
      break;
    }
    case Op::Eq:
    case Op::Ne:
    case Op::Gt:
    case Op::Lt:
    case Op::Le:
    case Op::Ge: {
      auto unbacked = [&](const Expr& e) {
        if (!is_symbol(e)) {
          return false;
        }
        const std::string& name = symbol_name(e);
        return !name.empty() && (name[0] == 'u' || name[0] == 'U');
      };
      bool evaluate =
          !((unbacked(a) && is_number(b)) || (unbacked(b) && is_number(a)));
      Rel kind = op == Op::Eq ? Rel::Eq
          : op == Op::Ne      ? Rel::Ne
          : op == Op::Gt      ? Rel::Gt
          : op == Op::Lt      ? Rel::Lt
          : op == Op::Le      ? Rel::Le
                              : Rel::Ge;
      out = rel(kind, a, b, evaluate);
      if (evaluate) {
        out = fold(out, env_->symbols());
      }
      break;
    }
    case Op::And:
      out = logical_and({a, b});
      break;
    case Op::Or:
      out = logical_or({a, b});
      break;
  }
  bool int_result = op <= Op::Max;
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, out, int_result ? PyType::Int : PyType::Bool, hint);
}

c10::SymNode NativeSymNodeImpl::unary(bool is_not, UnaryFn fallback) {
  if (proxy_mode()) {
    auto compute = [&]() -> c10::SymNode {
      if (!std::holds_alternative<std::monostate>(constant_) ||
          (is_not && torch_function_pending())) {
        return {};
      }
      auto lock = lock_env(*env_);
      return native_or_null([&] { return try_unary(is_not); });
    };
    return is_not
        ? proxy_dispatch("_sym_not", "sym_not", {clone()}, false, compute)
        : proxy_dispatch("_neg", "neg", {clone()}, false, compute);
  }
  {
    auto lock = lock_env(*env_);
    if (auto r = native_or_null([&] { return try_unary(is_not); })) {
      return r;
    }
  }
  return (materialize(*this).get()->*fallback)();
}

c10::SymNode NativeSymNodeImpl::try_unary(bool is_not) {
  PyType operand = is_not ? PyType::Bool : PyType::Int;
  if (!env_->replacements_empty() || pytype_ != operand) {
    return {};
  }
  Hint hint;
  Expr out;
  if (is_not) {
    if (has_hint()) {
      hint = !std::get<bool>(hint_);
    }
    out = logical_not(expr_);
  } else {
    if (has_hint()) {
      int64_t x = std::get<int64_t>(hint_);
      if (x == std::numeric_limits<int64_t>::min()) {
        return {};
      }
      hint = -x;
    }
    if (is_int_oo(expr_)) {
      return {};
    }
    out = -expr_;
  }
  return c10::make_intrusive<NativeSymNodeImpl>(env_, out, operand, hint);
}

#define NATIVE_BINARY(name, op)                                     \
  c10::SymNode NativeSymNodeImpl::name(const c10::SymNode& other) { \
    return binary(Op::op, &c10::SymNodeImpl::name, other);          \
  }
NATIVE_BINARY(add, Add)
NATIVE_BINARY(sub, Sub)
NATIVE_BINARY(mul, Mul)
NATIVE_BINARY(floordiv, FloorDiv)
NATIVE_BINARY(int_floordiv, FloorDiv)
NATIVE_BINARY(mod, Mod)
NATIVE_BINARY(pow_by_natural, PowByNatural)
NATIVE_BINARY(sym_min, Min)
NATIVE_BINARY(sym_max, Max)
NATIVE_BINARY(eq, Eq)
NATIVE_BINARY(ne, Ne)
NATIVE_BINARY(gt, Gt)
NATIVE_BINARY(lt, Lt)
NATIVE_BINARY(le, Le)
NATIVE_BINARY(ge, Ge)
NATIVE_BINARY(sym_and, And)
NATIVE_BINARY(sym_or, Or)
#undef NATIVE_BINARY

c10::SymNode NativeSymNodeImpl::neg() {
  return unary(false, &c10::SymNodeImpl::neg);
}

c10::SymNode NativeSymNodeImpl::sym_not() {
  return unary(true, &c10::SymNodeImpl::sym_not);
}

#define PYTHON_BINARY(name, py_method)                              \
  c10::SymNode NativeSymNodeImpl::name(const c10::SymNode& other) { \
    if (proxy_mode()) {                                             \
      return python_impl(py_method, {clone(), other});              \
    }                                                               \
    return materialize(*this)->name(to_python(other));              \
  }
PYTHON_BINARY(truediv, "_float_truediv")
PYTHON_BINARY(float_truediv, "_float_truediv")
PYTHON_BINARY(int_truediv, "_int_truediv")
PYTHON_BINARY(pow, "_float_pow")
PYTHON_BINARY(float_pow, "_float_pow")
#undef PYTHON_BINARY

#define PYTHON_UNARY(name, py_method)           \
  c10::SymNode NativeSymNodeImpl::name() {      \
    if (proxy_mode()) {                         \
      return python_impl(py_method, {clone()}); \
    }                                           \
    return materialize(*this)->name();          \
  }
PYTHON_UNARY(ceil, "_ceil")
PYTHON_UNARY(floor, "_floor")
PYTHON_UNARY(sym_float, "_sym_float")
#undef PYTHON_UNARY

c10::SymNode NativeSymNodeImpl::sym_ite(
    const c10::SymNode& then_val,
    const c10::SymNode& else_val) {
  if (proxy_mode()) {
    return python_impl("_sym_ite", {clone(), then_val, else_val});
  }
  return materialize(*this)->sym_ite(to_python(then_val), to_python(else_val));
}

c10::SymNode NativeSymNodeImpl::sizes_strides(
    SizesStrides fn,
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides) {
  static constexpr std::array py_methods = {
      "_is_contiguous",
      "_is_channels_last_contiguous_2d",
      "_is_channels_last_contiguous_3d",
      "_is_channels_last_strides_2d",
      "_is_channels_last_strides_3d",
      "_is_non_overlapping_and_dense_indicator"};
  static_assert(
      std::size(py_methods) ==
      static_cast<size_t>(SizesStrides::Indicator) + 1);
  const char* py_method = py_methods[static_cast<size_t>(fn)];
  if (proxy_mode()) {
    return python_impl(py_method, clone(), sizes, strides);
  }
  {
    auto lock = lock_env(*env_);
    if (auto r = native_or_null(
            [&] { return try_sizes_strides(fn, sizes, strides); })) {
      return r;
    }
  }
  return python_impl(
      py_method, materialize(*this), to_python(sizes), to_python(strides));
}

c10::SymNode NativeSymNodeImpl::try_sizes_strides(
    SizesStrides fn,
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides) {
  if (!env_->replacements_empty()) {
    return {};
  }
  // A missing hint makes Python compute one, which can populate caches.
  std::vector<Expr> exprs;
  std::vector<Expr> hints;
  for (auto nodes : {sizes, strides}) {
    for (const auto& n : nodes) {
      auto* o = as_native(n);
      if (o == nullptr || o->env_ != env_ || o->pytype_ != PyType::Int ||
          !o->has_hint() || is_int_oo(o->expr_)) {
        return {};
      }
      exprs.push_back(o->expr_);
      hints.emplace_back(std::get<int64_t>(o->hint_));
    }
  }
  const size_t dim = sizes.size();
  auto build = [&](c10::ArrayRef<Expr> args) -> Expr {
    auto size_args = args.slice(0, dim);
    auto stride_args = args.slice(dim);
    switch (fn) {
      case SizesStrides::Contiguous: {
        std::vector<size_t> order(dim);
        for (size_t i = 0; i < dim; ++i) {
          order[i] = dim - 1 - i;
        }
        return contiguous_generic(size_args, stride_args, order);
      }
      case SizesStrides::ChannelsLastContiguous2d:
        return contiguous_generic(size_args, stride_args, {1, 3, 2, 0});
      case SizesStrides::ChannelsLastContiguous3d:
        return contiguous_generic(size_args, stride_args, {1, 4, 3, 2, 0});
      case SizesStrides::ChannelsLastStrides2d:
        return channels_last_strides_generic(
            size_args, stride_args, {1, 3, 2, 0});
      case SizesStrides::ChannelsLastStrides3d:
        return channels_last_strides_generic(
            size_args, stride_args, {1, 4, 3, 2, 0});
      case SizesStrides::Indicator:
        break;
    }
    return function(Fn::IsNonOverlappingAndDenseIndicator, args);
  };
  if (strides.size() != dim) {
    throw NativeUnsupported("sizes and strides differ in length");
  }
  Expr out = build(exprs);
  // The hint is the same method applied to the int hints: bool(func(...)),
  // or eval_is_non_overlapping_and_dense for the indicator.
  Expr h = build(hints);
  if (fn == SizesStrides::Indicator) {
    auto i = as_int64(h);
    if (!i) {
      throw NativeUnsupported("indicator hint did not fold");
    }
    return c10::make_intrusive<NativeSymNodeImpl>(env_, out, PyType::Int, *i);
  }
  auto b = as_bool(h);
  if (!b) {
    throw NativeUnsupported("sizes/strides hint did not fold");
  }
  return c10::make_intrusive<NativeSymNodeImpl>(env_, out, PyType::Bool, *b);
}

#define NATIVE_SIZES_STRIDES(name, fn)                      \
  c10::SymNode NativeSymNodeImpl::name(                     \
      c10::ArrayRef<c10::SymNode> sizes,                    \
      c10::ArrayRef<c10::SymNode> strides) {                \
    return sizes_strides(SizesStrides::fn, sizes, strides); \
  }
NATIVE_SIZES_STRIDES(is_contiguous, Contiguous)
NATIVE_SIZES_STRIDES(is_channels_last_contiguous_2d, ChannelsLastContiguous2d)
NATIVE_SIZES_STRIDES(is_channels_last_contiguous_3d, ChannelsLastContiguous3d)
NATIVE_SIZES_STRIDES(is_channels_last_strides_2d, ChannelsLastStrides2d)
NATIVE_SIZES_STRIDES(is_channels_last_strides_3d, ChannelsLastStrides3d)
NATIVE_SIZES_STRIDES(is_non_overlapping_and_dense_indicator, Indicator)
#undef NATIVE_SIZES_STRIDES

c10::SymNode NativeSymNodeImpl::is_non_overlapping_and_dense(
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides) {
  // SymNode.is_non_overlapping_and_dense; wrap_int on the indicator is
  // wrap_int on self, and gives the eq a Python operand if the indicator is
  // Python.
  auto indicator = is_non_overlapping_and_dense_indicator(sizes, strides);
  return indicator->eq(indicator->wrap_int(1));
}

c10::SymNode NativeSymNodeImpl::sym_sum(c10::ArrayRef<c10::SymNode> args) {
  if (proxy_mode()) {
    return python_impl("sym_sum", clone(), args);
  }
  {
    auto lock = lock_env(*env_);
    if (auto r = native_or_null([&] { return try_sym_sum(args); })) {
      return r;
    }
  }
  return python_impl("sym_sum", materialize(*this), to_python(args));
}

c10::SymNode NativeSymNodeImpl::try_sym_sum(c10::ArrayRef<c10::SymNode> args) {
  if (!env_->replacements_empty()) {
    return {};
  }
  std::vector<Expr> exprs;
  std::optional<int64_t> total = 0;
  for (const auto& n : args) {
    auto* o = as_native(n);
    // int_oo is left to Python, as in try_binary.
    if (o == nullptr || o->env_ != env_ || o->pytype_ != PyType::Int ||
        is_int_oo(o->expr_)) {
      return {};
    }
    exprs.push_back(o->expr_);
    if (total) {
      if (!o->has_hint()) {
        total.reset();
      } else if (__builtin_add_overflow(
                     *total, std::get<int64_t>(o->hint_), &*total)) {
        return {};
      }
    }
  }
  Hint hint;
  if (total) {
    hint = *total;
  }
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, symbolic::add(exprs), PyType::Int, hint);
}

MaybeExpr NativeSymNodeImpl::evaluate(std::optional<bool> fallback_value) {
  if (!native_config_is_default()) {
    return std::nullopt;
  }
  auto lock = lock_env(*env_);
  return env_->evaluate_expr(expr_, hint_, fallback_value)
      .value_or(std::nullopt);
}

int64_t NativeSymNodeImpl::guard_int(const char* file, int64_t line) {
  MaybeExpr r = evaluate(std::nullopt);
  if (auto i = r ? as_int64(*r) : std::nullopt) {
    return *i;
  }
  return materialize(*this)->guard_int(file, line);
}

bool NativeSymNodeImpl::guard_bool(const char* file, int64_t line) {
  if (auto r = truth(evaluate(std::nullopt))) {
    return *r;
  }
  return materialize(*this)->guard_bool(file, line);
}

bool NativeSymNodeImpl::guard_or_false(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    if (auto r = truth(evaluate(false))) {
      return *r;
    }
  }
  return materialize(*this)->guard_or_false(file, line);
}

bool NativeSymNodeImpl::guard_or_true(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    if (auto r = truth(evaluate(true))) {
      return *r;
    }
  }
  return materialize(*this)->guard_or_true(file, line);
}

bool NativeSymNodeImpl::statically_known_true(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    // _sym_node_hint_disproves
    if (hint_ == Hint(false)) {
      return false;
    }
    std::optional<MaybeExpr> r;
    {
      auto lock = lock_env(*env_);
      r = env_->static_eval(expr_);
    }
    if (r) {
      return truth(*r).value_or(false);
    }
  }
  return materialize(*this)->statically_known_true(file, line);
}

bool NativeSymNodeImpl::expect_true(const char* file, int64_t line) {
  // A native env has prefer_deferred_runtime_asserts_over_guards unset, and
  // evaluate() answers only for mirrored (backed) symbols.
  if (has_hint()) {
    if (auto r = truth(evaluate(std::nullopt))) {
      return *r;
    }
  }
  return materialize(*this)->expect_true(file, line);
}

double NativeSymNodeImpl::guard_float(const char* file, int64_t line) {
  return materialize(*this)->guard_float(file, line);
}

bool NativeSymNodeImpl::guard_size_oblivious(const char* file, int64_t line) {
  return materialize(*this)->guard_size_oblivious(file, line);
}

int64_t NativeSymNodeImpl::int_() {
  return guard_int("", 0);
}

bool NativeSymNodeImpl::bool_() {
  return guard_bool("", 0);
}

} // namespace torch::symbolic
