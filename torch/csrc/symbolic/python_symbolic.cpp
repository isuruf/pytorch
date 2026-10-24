#include <torch/csrc/symbolic/python_symbolic.h>

#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/symbolic/Expr.h>
#include <torch/csrc/symbolic/Functions.h>
#include <torch/csrc/symbolic/NativeShapeEnv.h>
#include <torch/csrc/symbolic/NativeSymNodeImpl.h>
#include <torch/csrc/symbolic/PyFallback.h>
#include <torch/csrc/symbolic/ValueRanges.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <c10/core/impl/TorchDispatchModeTLS.h>

#include <symengine/add.h>
#include <symengine/infinity.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/visitor.h>

#include <array>
#include <atomic>
#include <optional>
#include <unordered_map>
#include <vector>

namespace torch::symbolic {

namespace py = pybind11;

namespace {

// Mirrors whether torch._ops' process-global pre-dispatch PROXY slot is set.
std::atomic<bool> pre_dispatch_proxy_set{false};

constexpr std::array kFunctions = {
    Fn::FloorDiv,
    Fn::CleanDiv,
    Fn::Mod,
    Fn::PythonMod,
    Fn::PowByNatural,
    Fn::IsNonOverlappingAndDenseIndicator,
};

int64_t to_int64(py::handle obj) {
  int overflow = 0;
  long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
  if (overflow != 0) {
    throw NativeUnsupported("integer overflow");
  }
  if (v == -1 && PyErr_Occurred()) {
    throw py::error_already_set();
  }
  return v;
}

// Python-side state for a SymbolTable: the sympy classes, the sympy Symbols
// that native symbols came from and a conversion cache.
struct PyArena {
  PyArena() {
    py::module_ sympy = py::module_::import("sympy");
    py::module_ numbers = py::module_::import("torch.utils._sympy.numbers");
    py::module_ functions = py::module_::import("torch.utils._sympy.functions");
    Integer = sympy.attr("Integer");
    Rational = sympy.attr("Rational");
    Symbol = sympy.attr("Symbol");
    Add = sympy.attr("Add");
    Mul = sympy.attr("Mul");
    Pow = sympy.attr("Pow");
    IntInfinity = numbers.attr("IntInfinity");
    NegativeIntInfinity = numbers.attr("NegativeIntInfinity");
    int_oo = numbers.attr("int_oo");
    oo = sympy.attr("oo");
    neg_oo = sympy.attr("S").attr("NegativeInfinity");
    true_ = sympy.attr("true");
    false_ = sympy.attr("false");
    Not = sympy.attr("Not");
    And = sympy.attr("And");
    Or = sympy.attr("Or");
    Eq = sympy.attr("Eq");
    Ne = sympy.attr("Ne");
    Lt = sympy.attr("Lt");
    Le = sympy.attr("Le");
    Gt = sympy.attr("Gt");
    Ge = sympy.attr("Ge");
    Max = functions.attr("Max");
    Min = functions.attr("Min");
    for (Fn fn : kFunctions) {
      function_classes.push_back(functions.attr(function_name(fn)));
    }
  }

  Expr from_sympy(py::handle obj);
  // An integer range bound: an Integer or +-int_oo (sympy's oo included).
  Expr bound_from_sympy(py::handle obj);
  ValueRanges range_from_sympy(py::handle lower, py::handle upper) {
    return ValueRanges(bound_from_sympy(lower), bound_from_sympy(upper));
  }
  py::object to_sympy(const Expr& e);
  py::object to_sympy(const MaybeExpr& e) {
    return e ? to_sympy(*e) : py::none();
  }

  c10::intrusive_ptr<SymbolTable> symbols = c10::make_intrusive<SymbolTable>();
  std::unordered_map<std::string, py::object> symbol_objects;
  ExprMap<py::object> sympy_cache;

  py::object Integer, Rational, Symbol, Add, Mul, Pow, IntInfinity,
      NegativeIntInfinity, int_oo, oo, neg_oo, true_, false_, Not, And, Or, Eq,
      Ne, Lt, Le, Gt, Ge, Max, Min;
  // Parallel to kFunctions.
  std::vector<py::object> function_classes;
};

// Builds the sympy expression of a SymEngine expression. bvisit is how
// SymEngine's BaseVisitor dispatches, not a misspelled override of visit.
// NOLINTBEGIN(bugprone-virtual-near-miss)
class ToSympy : public SymEngine::BaseVisitor<ToSympy> {
 public:
  explicit ToSympy(PyArena& arena) : a_(&arena) {}

  py::object apply(const Expr& e) {
    auto it = a_->sympy_cache.find(e);
    if (it != a_->sympy_cache.end()) {
      return it->second;
    }
    e.get_basic()->accept(*this);
    py::object r = std::move(result_);
    a_->sympy_cache.emplace(e, r);
    return r;
  }

  void bvisit(const SymEngine::Basic& x) {
    throw NativeUnsupported("no sympy conversion for " + x.__str__());
  }
  void bvisit(const SymEngine::Integer& x) {
    result_ = a_->Integer(int_to_py(x.as_integer_class()));
  }
  void bvisit(const SymEngine::Rational& x) {
    result_ = a_->Rational(
        int_to_py(x.get_num()->as_integer_class()),
        int_to_py(x.get_den()->as_integer_class()));
  }
  void bvisit(const SymEngine::Infty& x) {
    result_ = x.is_positive() ? a_->int_oo : a_->int_oo.attr("__neg__")();
  }
  void bvisit(const SymEngine::Symbol& x) {
    auto it = a_->symbol_objects.find(x.get_name());
    if (it != a_->symbol_objects.end()) {
      result_ = it->second;
      return;
    }
    // A symbol made natively; its facts are the sympy assumptions.
    const SymbolFacts* f = a_->symbols->facts(Expr(x.rcp_from_this()));
    py::dict assumptions;
    if (f != nullptr) {
      for (auto [name, v] :
           {std::pair{"integer", f->integer},
            std::pair{"nonnegative", f->nonnegative},
            std::pair{"positive", f->positive}}) {
        if (v) {
          assumptions[name] = true;
        }
      }
    }
    result_ = a_->Symbol(x.get_name(), **assumptions);
    a_->symbol_objects.emplace(x.get_name(), result_);
  }
  void bvisit(const SymEngine::Add& x) {
    result_ = call(a_->Add, x);
  }
  void bvisit(const SymEngine::Mul& x) {
    result_ = call(a_->Mul, x);
  }
  void bvisit(const SymEngine::Pow& x) {
    result_ = call(a_->Pow, x);
  }
  void bvisit(const SymEngine::BooleanAtom& x) {
    result_ = x.get_val() ? a_->true_ : a_->false_;
  }
  void bvisit(const SymEngine::Equality& x) {
    result_ = call(a_->Eq, x, /*evaluate=*/false);
  }
  void bvisit(const SymEngine::Unequality& x) {
    result_ = call(a_->Ne, x, /*evaluate=*/false);
  }
  // SymEngine keeps a > b as b < a; a number goes back on the right.
  void bvisit(const SymEngine::LessThan& x) {
    result_ = number_first(x) ? call(a_->Ge, x, false, true)
                              : call(a_->Le, x, /*evaluate=*/false);
  }
  void bvisit(const SymEngine::StrictLessThan& x) {
    result_ = number_first(x) ? call(a_->Gt, x, false, true)
                              : call(a_->Lt, x, /*evaluate=*/false);
  }
  void bvisit(const SymEngine::Not& x) {
    result_ = call(a_->Not, x);
  }
  void bvisit(const SymEngine::And& x) {
    result_ = call(a_->And, x);
  }
  void bvisit(const SymEngine::Or& x) {
    result_ = call(a_->Or, x);
  }
  void bvisit(const SymEngine::Max& x) {
    result_ = call(a_->Max, x, /*evaluate=*/false);
  }
  void bvisit(const SymEngine::Min& x) {
    result_ = call(a_->Min, x, /*evaluate=*/false);
  }
  void bvisit(const SymEngine::FunctionWrapper& x) {
    const auto* f = dynamic_cast<const TorchFunction*>(&x);
    if (f == nullptr) {
      bvisit(static_cast<const SymEngine::Basic&>(x));
      return;
    }
    size_t i = static_cast<size_t>(f->fn());
    result_ = call(a_->function_classes.at(i), x, /*evaluate=*/false);
  }

 private:
  static py::object int_to_py(const SymEngine::integer_class& i) {
    if (SymEngine::mp_fits_slong_p(i)) {
      return py::int_(SymEngine::mp_get_si(i));
    }
    std::string s = SymEngine::mp_get_hex_str(i);
    PyObject* r = PyLong_FromString(s.c_str(), nullptr, 16);
    if (r == nullptr) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(r);
  }

  static bool number_first(const SymEngine::Basic& x) {
    const auto& xs = x.get_args();
    return is_number(Expr(xs[0])) && !is_number(Expr(xs[1]));
  }

  py::object call(
      const py::object& cls,
      const SymEngine::Basic& x,
      std::optional<bool> evaluate = std::nullopt,
      bool reversed = false) {
    const auto& xs = x.get_args();
    py::tuple args(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      args[reversed ? xs.size() - 1 - i : i] = apply(Expr(xs[i]));
    }
    if (evaluate) {
      return cls(*args, py::arg("evaluate") = *evaluate);
    }
    return cls(*args);
  }

  PyArena* a_;
  py::object result_;
};
// NOLINTEND(bugprone-virtual-near-miss)

py::object PyArena::to_sympy(const Expr& e) {
  return ToSympy(*this).apply(e);
}

Expr PyArena::from_sympy(py::handle obj) {
  auto from_args = [&]() {
    std::vector<Expr> args;
    for (py::handle a : obj.attr("args")) {
      args.push_back(from_sympy(a));
    }
    return args;
  };
  auto exact = [&](const py::object& cls) {
    return Py_TYPE(obj.ptr()) == reinterpret_cast<PyTypeObject*>(cls.ptr());
  };
  if (PyLong_CheckExact(obj.ptr()) || py::isinstance(obj, Integer)) {
    return Expr(to_int64(py::int_(py::reinterpret_borrow<py::object>(obj))));
  }
  if (py::isinstance(obj, IntInfinity)) {
    return symbolic::int_oo();
  }
  if (py::isinstance(obj, NegativeIntInfinity)) {
    return symbolic::neg_int_oo();
  }
  if (py::isinstance(obj, Rational)) {
    return Expr(SymEngine::Rational::from_two_ints(
        to_int64(obj.attr("p")), to_int64(obj.attr("q"))));
  }
  // Dummy and Wild compare by more than the name.
  if (exact(Symbol)) {
    py::dict assumptions = obj.attr("assumptions0");
    auto fact = [&](const char* name) {
      return assumptions.contains(name) && assumptions[name].cast<bool>();
    };
    auto name = obj.attr("name").cast<std::string>();
    Expr e = symbols->symbol(
        name, {fact("integer"), fact("nonnegative"), fact("positive")});
    symbol_objects.emplace(name, py::reinterpret_borrow<py::object>(obj));
    return e;
  }
  if (py::isinstance(obj, Add)) {
    return add(from_args());
  }
  if (py::isinstance(obj, Mul)) {
    return mul(from_args());
  }
  if (py::isinstance(obj, Pow)) {
    auto a = from_args();
    return SymEngine::pow(a[0], a[1]);
  }
  if (obj.is(true_) || obj.is(false_)) {
    return boolean(obj.is(true_));
  }
  for (auto [cls, kind] :
       {std::pair{&Eq, Rel::Eq},
        std::pair{&Ne, Rel::Ne},
        std::pair{&Lt, Rel::Lt},
        std::pair{&Le, Rel::Le},
        std::pair{&Gt, Rel::Gt},
        std::pair{&Ge, Rel::Ge}}) {
    if (exact(*cls)) {
      auto a = from_args();
      return rel(kind, a[0], a[1], /*evaluate=*/false);
    }
  }
  if (exact(Not)) {
    return logical_not(from_args()[0]);
  }
  if (exact(And)) {
    return logical_and(from_args());
  }
  if (exact(Or)) {
    return logical_or(from_args());
  }
  if (exact(Max)) {
    return sym_max(from_args());
  }
  if (exact(Min)) {
    return sym_min(from_args());
  }
  for (size_t i = 0; i < function_classes.size(); ++i) {
    // Exact type: CleanDiv subclasses FloorDiv.
    if (exact(function_classes[i])) {
      return function(kFunctions[i], from_args());
    }
  }
  throw NativeUnsupported(
      "unsupported sympy type " +
      py::str(py::type::handle_of(obj).attr("__name__")).cast<std::string>());
}

Expr PyArena::bound_from_sympy(py::handle obj) {
  if (obj.is(oo) || py::isinstance(obj, IntInfinity)) {
    return symbolic::int_oo();
  }
  if (obj.is(neg_oo) || py::isinstance(obj, NegativeIntInfinity)) {
    return symbolic::neg_int_oo();
  }
  Expr e = from_sympy(obj);
  if (!is_integer(e)) {
    throw NativeUnsupported("range bound that is not an integer");
  }
  return e;
}

// NativeShapeEnv::binding().
struct EnvBinding {
  std::shared_ptr<PyArena> arena;
  // A weakref to the ShapeEnv, or None. The ShapeEnv owns the native env.
  py::object shape_env;
  // The ShapeEnv while the env has live nodes, else None.
  py::object live_shape_env = py::none();
};

struct PyShapeEnv {
  PyShapeEnv(std::shared_ptr<PyArena> owner, const py::object& shape_env)
      : owner(std::move(owner)),
        env(c10::make_intrusive<NativeShapeEnv>(this->owner->symbols)) {
    auto* binding = new EnvBinding{
        this->owner,
        shape_env.is_none() ? py::object(py::none())
                            : py::object(py::weakref(shape_env))};
    // Native nodes may drop the env without the GIL.
    env->set_binding(std::shared_ptr<EnvBinding>(binding, [](EnvBinding* b) {
      if (!Py_IsInitialized()) {
        return;
      }
      py::gil_scoped_acquire gil;
      delete b;
    }));
  }

  std::shared_ptr<PyArena> owner;
  c10::intrusive_ptr<NativeShapeEnv> env;
};

// A None, int or bool hint; nullopt for any other, or an int that does not
// fit.
std::optional<Hint> hint_from_py(py::handle hint) {
  if (PyBool_Check(hint.ptr())) {
    return Hint(hint.ptr() == Py_True);
  }
  if (PyLong_CheckExact(hint.ptr())) {
    int overflow = 0;
    int64_t v = PyLong_AsLongLongAndOverflow(hint.ptr(), &overflow);
    if (overflow != 0) {
      return std::nullopt;
    }
    return Hint(v);
  }
  if (hint.is_none()) {
    return Hint();
  }
  return std::nullopt;
}

// A torch.fx.experimental._config entry, read the way ConfigModule.__getattr__
// resolves an entry without alias or justknob, but without running Python
// code. Everything but `hide` and the user override is fixed at creation.
struct ConfigEntry {
  ConfigEntry(py::handle entry, py::handle unset)
      : entry(py::reinterpret_borrow<py::object>(entry)),
        user_override(entry.attr("user_override")),
        forced(entry.attr("env_value_force")) {
    plain = entry.attr("alias").is_none() && entry.attr("justknob").is_none();
    fallback = entry.attr("env_value_default");
    if (fallback.is(unset)) {
      fallback = entry.attr("default");
    }
  }

  // Aliased, justknob and hidden entries count as set.
  bool falsy(PyObject* unset, PyObject* hide_str) const {
    if (!plain) {
      return false;
    }
    py::object hidden = py::reinterpret_steal<py::object>(
        PyObject_GetAttr(entry.ptr(), hide_str));
    if (!hidden) {
      throw py::error_already_set();
    }
    int r = PyObject_IsTrue(hidden.ptr());
    if (r != 0) {
      if (r < 0) {
        throw py::error_already_set();
      }
      return false;
    }
    PyObject* value = forced.ptr();
    py::object override_value;
    if (value == unset) {
      if (PyContextVar_Get(user_override.ptr(), nullptr, &value) != 0) {
        throw py::error_already_set();
      }
      override_value = py::reinterpret_steal<py::object>(value);
    }
    if (value == unset) {
      value = fallback.ptr();
    }
    r = PyObject_IsTrue(value);
    if (r < 0) {
      throw py::error_already_set();
    }
    return r == 0;
  }

  py::object entry;
  py::object user_override;
  py::object forced;
  py::object fallback;
  bool plain;
};

EnvBinding& binding_of(NativeShapeEnv& env) {
  auto* b = static_cast<EnvBinding*>(env.binding());
  TORCH_CHECK(b != nullptr, "native env without a Python binding");
  return *b;
}

py::object pytype_to_py(PyType t) {
  return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
      t == PyType::Int ? &PyLong_Type : &PyBool_Type));
}

// Puts the native mod choices where binary_magic_impl would have cached them.
// Runs before the env leaves pristine and before _symop_cache is copied.
void flush_mod_memo(PyShapeEnv& self) {
  const py::object& ref = binding_of(*self.env).shape_env;
  if (!self.env->pristine() || ref.is_none()) {
    return;
  }
  py::object shape_env = ref();
  if (shape_env.is_none()) {
    return;
  }
  PyArena& a = *self.owner;
  py::object cache = shape_env.attr("_symop_cache");
  py::object version = shape_env.attr("_replacements_version_counter");
  for (const auto& [key, out] : self.env->mod_memo()) {
    cache.attr("setdefault")(
        py::make_tuple(
            "mod", a.to_sympy(key.first), a.to_sympy(key.second), version),
        py::make_tuple(a.to_sympy(out), pytype_to_py(PyType::Int), false));
  }
}

py::object node_expr(const NativeSymNodeImpl& node) {
  NativeShapeEnv& env = *node.env();
  auto lock = lock_env(env);
  return binding_of(env).arena->to_sympy(node.expr());
}

c10::SymNode node_from_py(py::handle obj) {
  if (py::isinstance<c10::SymNodeImpl>(obj)) {
    return py::cast<c10::SymNode>(obj);
  }
  return c10::make_intrusive<impl::PythonSymNodeImpl>(
      py::reinterpret_borrow<py::object>(obj));
}

// sympy's node.expr.<attr>, computed by `native` from the native expr unless
// replacements could change node.expr.
template <typename F>
bool expr_query(const NativeSymNodeImpl& node, const char* attr, F native) {
  {
    NativeShapeEnv& env = *node.env();
    auto lock = lock_env(env);
    if (env.replacements_empty()) {
      return native(node.expr());
    }
  }
  return node_to_py(materialize(node)).attr("expr").attr(attr).cast<bool>();
}

std::vector<c10::SymNode> nodes_from_py(const py::sequence& nodes) {
  std::vector<c10::SymNode> r;
  r.reserve(nodes.size());
  for (py::handle n : nodes) {
    r.push_back(node_from_py(n));
  }
  return r;
}

py::list nodes_to_py(c10::ArrayRef<c10::SymNode> nodes) {
  py::list r(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    r[i] = node_to_py(nodes[i]);
  }
  return r;
}

py::handle python_symnode_class() {
  static py::handle cls =
      py::object(
          py::module_::import("torch.fx.experimental.sym_node").attr("SymNode"))
          .release();
  return cls;
}

} // namespace

py::object node_to_py(const c10::SymNode& n) {
  if (auto* p = dynamic_cast<impl::PythonSymNodeImpl*>(n.get())) {
    return py::reinterpret_borrow<py::object>(p->getPyObj());
  }
  return py::cast(n);
}

bool native_config_is_default() {
  std::optional<py::gil_scoped_acquire> gil;
  if (py::detail::get_thread_state_unchecked() == nullptr) {
    gil.emplace();
  }
  struct Entries {
    py::object unset;
    py::object hide_str;
    ConfigEntry backed;
    ConfigEntry aggressive;
  };
  static const auto* entries = [] {
    py::dict config =
        py::module_::import("torch.fx.experimental._config").attr("_config");
    py::object unset = py::module_::import("torch.utils._config_module")
                           .attr("_UNSET_SENTINEL");
    return new Entries{
        unset,
        py::reinterpret_steal<py::object>(PyUnicode_InternFromString("hide")),
        ConfigEntry(config["backed_size_oblivious"], unset),
        ConfigEntry(config["aggressive_guard_free_semantics"], unset)};
  }();
  PyObject* unset = entries->unset.ptr();
  PyObject* hide_str = entries->hide_str.ptr();
  return entries->backed.falsy(unset, hide_str) &&
      entries->aggressive.falsy(unset, hide_str);
}

std::string sympy_str(const NativeSymNodeImpl& node) {
  py::gil_scoped_acquire gil;
  return py::str(node_expr(node)).cast<std::string>();
}

void live_nodes_changed(NativeShapeEnv& env) {
  if (!Py_IsInitialized()) {
    return;
  }
  py::gil_scoped_acquire gil;
  auto* b = static_cast<EnvBinding*>(env.binding());
  if (b == nullptr || b->shape_env.is_none()) {
    return;
  }
  // Re-read under the GIL: concurrent transitions converge on the last one.
  if (env.live_nodes() == 0) {
    b->live_shape_env = py::none();
  } else if (b->live_shape_env.is_none()) {
    b->live_shape_env = b->shape_env();
  }
}

std::unique_lock<std::mutex> lock_env(NativeShapeEnv& env) {
  std::unique_lock<std::mutex> lock(env.mutex(), std::try_to_lock);
  if (!lock.owns_lock()) {
    if (PyGILState_Check()) {
      py::gil_scoped_release no_gil;
      lock.lock();
    } else {
      lock.lock();
    }
  }
  return lock;
}

c10::SymNode materialize(const NativeSymNodeImpl& node) {
  py::gil_scoped_acquire gil;
  EnvBinding& binding = binding_of(*node.env());
  py::object shape_env =
      binding.shape_env.is_none() ? py::none() : binding.shape_env();
  TORCH_CHECK(!shape_env.is_none(), "the ShapeEnv of a native SymNode is gone");
  py::module_ sym_node = py::module_::import("torch.fx.experimental.sym_node");
  py::object hint = std::holds_alternative<std::monostate>(node.hint())
      ? py::object(sym_node.attr("_NO_HINT"))
      : hint_to_py(node.hint());
  py::object constant = hint_to_py(node.constant());
  py::object r = sym_node.attr("SymNode")(
      node_expr(node),
      shape_env,
      pytype_to_py(node.pytype()),
      hint,
      py::arg("constant") = constant,
      py::arg("fx_node") = constant);
  return c10::make_intrusive<impl::PythonSymNodeImpl>(std::move(r));
}

bool proxy_mode() {
  return pre_dispatch_proxy_set.load(std::memory_order_relaxed) ||
      c10::impl::TorchDispatchModeTLS::get_mode(
          c10::impl::TorchDispatchModeKey::PROXY)
          .has_value();
}

c10::SymNode python_impl(const char* method, c10::ArrayRef<c10::SymNode> args) {
  py::gil_scoped_acquire gil;
  py::tuple py_args(nodes_to_py(args));
  return node_from_py(python_symnode_class().attr(method)(*py_args));
}

namespace {

// proxy_tensor._sym_register with thunkify off: _compute_proxy, then
// set_proxy_slot. Calls _sym_register for a lazy thunk, a Symbol result,
// set_proxy_slot debug logging, or an operand not in symnode_tracker.
void sym_register(
    const py::object& proxy_tensor,
    py::handle tracer,
    const py::object& op,
    const py::tuple& wrapped,
    const py::object& sym,
    const NativeSymNodeImpl& out) {
  static const auto* statics = new std::array<py::object, 5>{
      proxy_tensor.attr("log"),
      py::module_::import("logging").attr("DEBUG"),
      proxy_tensor.attr("py_sym_types"),
      proxy_tensor.attr("fx").attr("Proxy"),
      proxy_tensor.attr("Thunk")};
  const auto& [log, debug, sym_types, proxy_type, thunk_type] = *statics;
  auto python = [&] {
    proxy_tensor.attr("_sym_register")(tracer, op, wrapped, sym);
  };
  if (is_symbol(out.expr()) || tracer.attr("enable_thunkify").cast<bool>() ||
      log.attr("isEnabledFor")(debug).cast<bool>()) {
    return python();
  }
  py::dict nodes = tracer.attr("symnode_tracker").attr("sym_node_dict");
  py::tuple n_args(wrapped.size());
  for (size_t i = 0; i < wrapped.size(); ++i) {
    py::handle a = wrapped[i];
    int is_sym = PyObject_IsInstance(a.ptr(), sym_types.ptr());
    if (is_sym < 0) {
      throw py::error_already_set();
    }
    if (is_sym == 0) {
      n_args[i] = a;
      continue;
    }
    PyObject* thunk =
        PyDict_GetItemWithError(nodes.ptr(), a.attr("node").ptr());
    if (thunk == nullptr) {
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
      return python();
    }
    n_args[i] = py::handle(thunk).attr("force")().attr("node");
  }
  py::object n_out =
      tracer.attr("create_node")("call_function", op, n_args, py::dict());
  py::object p_out = proxy_type(n_out, tracer);
  n_out.attr("meta")["val"] = sym;
  py::object thunk = thunk_type(py::none());
  thunk.attr("r") = p_out;
  py::object key = sym.attr("node");
  if (!nodes.contains(key)) {
    nodes[key] = thunk;
  }
}

} // namespace

c10::SymNode proxy_dispatch(
    const char* method,
    const char* op_name,
    c10::ArrayRef<c10::SymNode> args,
    bool mul,
    c10::function_ref<c10::SymNode()> compute) {
  py::gil_scoped_acquire gil;
  static const auto* modules = new std::array<py::object, 3>{
      py::module_::import("torch._logging._internal"),
      py::module_::import("torch.fx.experimental.sym_node"),
      py::module_::import("torch.fx.experimental.proxy_tensor")};
  const auto& [logging, sym_node, proxy_tensor] = *modules;
  if (logging.attr("GET_DTRACE_STRUCTURED").cast<bool>()) {
    return python_impl(method, args);
  }
  py::tuple wrapped(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto* n = static_cast<NativeSymNodeImpl*>(args[i].get());
    if (!std::holds_alternative<std::monostate>(n->constant())) {
      wrapped[i] = hint_to_py(n->constant());
    } else {
      wrapped[i] = make_sym_object(
          n->is_int() ? get_symint_class() : get_symbool_class(),
          node_to_py(args[i]));
    }
  }
  py::object op = sym_node.attr("METHOD_TO_OPERATOR")[op_name];
  py::object r;
  static const auto* proxy_mode_type =
      new py::object(proxy_tensor.attr("ProxyTorchDispatchMode"));
  static PyObject* const tracer_str = PyUnicode_InternFromString("tracer");
  auto mode = pre_dispatch_proxy_set.load(std::memory_order_relaxed)
      ? std::nullopt
      : c10::impl::TorchDispatchModeTLS::get_mode(
            c10::impl::TorchDispatchModeKey::PROXY);
  PyObject* mode_obj =
      mode.has_value() ? (*mode)->ptr(getPyInterpreter()) : nullptr;
  if (mode_obj != nullptr &&
      Py_TYPE(mode_obj) ==
          reinterpret_cast<PyTypeObject*>(proxy_mode_type->ptr()) &&
      sym_glue_native()) {
    auto is_one = [](const c10::SymNode& n) {
      const Hint& c = static_cast<NativeSymNodeImpl*>(n.get())->constant();
      const int64_t* i = std::get_if<int64_t>(&c);
      const bool* b = std::get_if<bool>(&c);
      return (i != nullptr && *i == 1) || (b != nullptr && *b);
    };
    if (mul && is_one(args[1])) {
      r = wrapped[0];
    } else if (mul && is_one(args[0])) {
      r = wrapped[1];
    } else if (c10::SymNode out = compute()) {
      py::object sym = make_sym_object(
          out->is_int() ? get_symint_class() : get_symbool_class(),
          node_to_py(out));
      sym_register(
          proxy_tensor,
          py::handle(mode_obj).attr(tracer_str),
          op,
          wrapped,
          sym,
          *static_cast<NativeSymNodeImpl*>(out.get()));
      return out;
    }
  }
  if (!r) {
    r = proxy_tensor.attr("handle_sym_dispatch")(op, wrapped, py::dict());
  }
  if (is_symint(r) || is_symfloat(r) || is_symbool(r)) {
    return node_from_py(r.attr("node"));
  }
  PyObject* t = reinterpret_cast<PyObject*>(Py_TYPE(r.ptr()));
  const char* wrap = t == reinterpret_cast<PyObject*>(&PyBool_Type)
      ? "wrap_bool"
      : t == reinterpret_cast<PyObject*>(&PyLong_Type)  ? "wrap_int"
      : t == reinterpret_cast<PyObject*>(&PyFloat_Type) ? "wrap_float"
                                                        : nullptr;
  if (wrap == nullptr) {
    return node_from_py(py::handle(Py_NotImplemented));
  }
  return node_from_py(node_to_py(args[0]).attr(wrap)(r));
}

c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides) {
  py::gil_scoped_acquire gil;
  return node_from_py(python_symnode_class().attr(method)(
      node_to_py(self), nodes_to_py(sizes), nodes_to_py(strides)));
}

c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> args) {
  py::gil_scoped_acquire gil;
  return node_from_py(
      python_symnode_class().attr(method)(node_to_py(self), nodes_to_py(args)));
}

void initSymbolicBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();
  auto sm = m.def_submodule("_symbolic", "native symbolic expressions");
  py::register_exception<NativeUnsupported>(sm, "NativeUnsupported");
  initGlueBindings(sm);
  sm.def("_native_config_is_default", &native_config_is_default);
  sm.def("_native_queries_pending", &NativeShapeEnv::queries_pending);
  sm.def("_set_suppress_guards", [](bool v) { suppress_guards_tls() = v; });
  sm.def("_suppress_guards", [] { return suppress_guards_tls(); });
  sm.def("_set_pre_dispatch_proxy", [](bool v) { pre_dispatch_proxy_set = v; });
  sm.def("_roundtrip_symbool", [](const c10::SymBool& v) { return v; });
  sm.def("_roundtrip_symfloat", [](const c10::SymFloat& v) { return v; });

  // Takes and returns sympy expressions; the native ones stay internal.
  using Self = std::shared_ptr<PyArena>;
  py::class_<PyArena, Self>(sm, "_Arena")
      .def(py::init<>())
      .def(
          "roundtrip",
          [](const Self& self, py::handle e) {
            return self->to_sympy(self->from_sympy(e));
          })
      .def(
          "sstr",
          [](const Self& self, py::handle e) {
            return SymEngine::str(*self->from_sympy(e).get_basic());
          })
      .def(
          "function",
          [](const Self& self, const std::string& name, const py::list& args) {
            auto fn = function_from_name(name);
            TORCH_CHECK(fn, "unknown function ", name);
            std::vector<Expr> a;
            for (py::handle x : args) {
              a.push_back(self->from_sympy(x));
            }
            return self->to_sympy(function(*fn, a));
          })
      .def(
          "bound_sympy",
          [](const Self& self, py::handle e, const py::dict& ranges) {
            RangeMap m;
            for (auto [sym, range] : ranges) {
              auto bounds = range.cast<py::tuple>();
              m.insert_or_assign(
                  self->from_sympy(sym),
                  self->range_from_sympy(bounds[0], bounds[1]));
            }
            ValueRanges r = bound_sympy(self->from_sympy(e), m, *self->symbols);
            return py::make_tuple(
                self->to_sympy(r.lower), self->to_sympy(r.upper));
          });

  py::class_<PyShapeEnv>(sm, "NativeShapeEnv")
      .def(
          py::init<Self, const py::object&>(),
          py::arg("arena"),
          py::arg("shape_env") = py::none())
      .def_property_readonly(
          "arena", [](const PyShapeEnv& self) { return self.owner; })
      .def_property_readonly(
          "pristine",
          [](const PyShapeEnv& self) { return self.env->pristine(); })
      .def_property_readonly(
          "replacements_empty",
          [](const PyShapeEnv& self) { return self.env->replacements_empty(); })
      .def(
          "mirror_symbol",
          [](PyShapeEnv& self,
             py::handle sym,
             py::handle hint,
             py::handle lower,
             py::handle upper) {
            // A symbol that is not representable stays unmirrored, so every
            // query mentioning it delegates.
            PyArena& a = *self.owner;
            auto lock = lock_env(*self.env);
            try {
              self.env->add_symbol(
                  a.from_sympy(sym),
                  hint.is_none() ? std::nullopt
                                 : std::optional<int64_t>(to_int64(hint)),
                  a.range_from_sympy(lower, upper));
            } catch (const NativeUnsupported&) {
              return;
            }
          },
          py::arg("sym"),
          py::arg("hint"),
          py::arg("lower"),
          py::arg("upper"))
      .def(
          "mirrored",
          [](PyShapeEnv& self, py::handle sym) -> py::object {
            std::optional<std::pair<std::optional<int64_t>, ValueRanges>> m;
            auto lock = lock_env(*self.env);
            try {
              m = self.env->mirrored(self.owner->from_sympy(sym));
            } catch (const NativeUnsupported&) {
              return py::none();
            }
            if (!m) {
              return py::none();
            }
            auto& [hint, range] = *m;
            return py::make_tuple(
                hint,
                self.owner->to_sympy(range.lower),
                self.owner->to_sympy(range.upper),
                false);
          })
      // A copied or unpickled ShapeEnv gets no native env.
      .def(
          "__deepcopy__",
          [](const PyShapeEnv&, const py::dict&) { return py::none(); })
      .def(
          "__reduce__",
          [](const PyShapeEnv&) {
            return py::make_tuple(py::type::of(py::none()), py::tuple());
          })
      .def(
          "update_range",
          [](PyShapeEnv& self,
             py::handle sym,
             py::handle lower,
             py::handle upper) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->update_range(
                self.owner->from_sympy(sym),
                self.owner->range_from_sympy(lower, upper));
          })
      .def(
          "mark_not_pristine",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->mark_not_pristine();
          })
      .def(
          "mark_replacements",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->mark_replacements();
          })
      .def(
          "flush_mod_memo",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
          })
      .def(
          "simplify",
          [](PyShapeEnv& self, py::handle e) {
            auto lock = lock_env(*self.env);
            return self.owner->to_sympy(
                self.env->simplify(self.owner->from_sympy(e)));
          })
      .def(
          "maybe_evaluate_static",
          [](PyShapeEnv& self, py::handle e) {
            auto lock = lock_env(*self.env);
            return self.owner->to_sympy(
                self.env->maybe_evaluate_static(self.owner->from_sympy(e)));
          })
      .def(
          "static_eval",
          [](PyShapeEnv& self, py::handle e) {
            // (answered, result).
            auto lock = lock_env(*self.env);
            auto r = self.env->static_eval(self.owner->from_sympy(e));
            return py::make_tuple(
                r.has_value(), self.owner->to_sympy(r.value_or(std::nullopt)));
          })
      .def(
          "evaluate_expr",
          [](PyShapeEnv& self,
             py::handle e,
             py::handle hint,
             std::optional<bool> fallback_value) -> py::object {
            // The result, or None when not answered natively; only None, int
            // and bool hints are native.
            auto h = hint_from_py(hint);
            if (!h || !native_config_is_default()) {
              return py::none();
            }
            auto lock = lock_env(*self.env);
            auto r = self.env->evaluate_expr(
                self.owner->from_sympy(e), *h, fallback_value);
            return self.owner->to_sympy(r.value_or(std::nullopt));
          },
          py::arg("e"),
          py::arg("hint") = py::none(),
          py::arg("fallback_value") = py::none())
      .def(
          "make_node",
          [](PyShapeEnv& self,
             py::handle expr,
             py::handle pytype,
             py::handle hint) {
            bool is_bool = pytype.is(pytype_to_py(PyType::Bool));
            TORCH_CHECK(
                is_bool || pytype.is(pytype_to_py(PyType::Int)),
                "native nodes are int or bool");
            TORCH_CHECK(
                hint.is_none() ||
                    (is_bool ? PyBool_Check(hint.ptr())
                             : PyLong_CheckExact(hint.ptr())),
                "hint must be None or of the pytype");
            auto h = hint_from_py(hint);
            if (!h) {
              throw NativeUnsupported("hint overflows int64");
            }
            auto lock = lock_env(*self.env);
            Expr e = self.owner->from_sympy(expr);
            return c10::make_intrusive<NativeSymNodeImpl>(
                self.env, e, is_bool ? PyType::Bool : PyType::Int, *h);
          })
      .def("take_queries", [](PyShapeEnv& self) {
        // (seq, expr, evaluate, hint, fallback_value, suppress_guards,
        // result) with sympy exprs.
        PyArena& a = *self.owner;
        py::list out;
        auto lock = lock_env(*self.env);
        for (const auto& [seq, q, result] : self.env->take_queries()) {
          out.append(py::make_tuple(
              seq,
              a.to_sympy(q.expr),
              q.evaluate,
              hint_to_py(q.hint),
              q.fallback_value,
              q.suppress_guards,
              a.to_sympy(result)));
        }
        return out;
      });

  // Unlike the _SymNode methods it overrides, these take and return Python
  // SymNodes as well.
  py::class_<
      NativeSymNodeImpl,
      c10::SymNodeImpl,
      c10::intrusive_ptr<NativeSymNodeImpl>>
      node_cls(sm, "_NativeSymNode");
  node_cls
      .def_property_readonly(
          "_expr", [](const NativeSymNodeImpl& n) { return node_expr(n); })
      .def_property_readonly(
          "expr",
          [](const NativeSymNodeImpl& n) -> py::object {
            bool replaced = false;
            {
              auto lock = lock_env(*n.env());
              replaced = !n.env()->replacements_empty();
            }
            if (!replaced) {
              return node_expr(n);
            }
            return node_to_py(materialize(n)).attr("expr");
          })
      .def_property_readonly(
          "_expr_is_number",
          [](const NativeSymNodeImpl& n) {
            return expr_query(n, "is_number", [](const Expr& e) {
              return !is_boolean(e) && free_symbols(e).empty();
            });
          })
      .def_property_readonly(
          "_expr_is_Boolean",
          [](const NativeSymNodeImpl& n) {
            // BooleanAtom and BooleanFunction; Relational is not.
            return expr_query(n, "is_Boolean", [](const Expr& e) {
              return is_boolean(e) && !is_relational(e);
            });
          })
      .def_property_readonly(
          "hint",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.hint()); })
      .def_property_readonly(
          "_hint",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.hint()); })
      .def_property_readonly(
          "shape_env",
          [](const NativeSymNodeImpl& n) -> py::object {
            const py::object& ref = binding_of(*n.env()).shape_env;
            return ref.is_none() ? py::none() : ref();
          })
      // Native envs exist only without translation validation.
      .def_property_readonly(
          "fx_node", [](const NativeSymNodeImpl& /*n*/) { return false; })
      .def_property_readonly(
          "constant",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.constant()); })
      .def_property_readonly(
          "pytype",
          [](const NativeSymNodeImpl& n) { return pytype_to_py(n.pytype()); })
      .def_property_readonly(
          "_optimized_summation",
          [](const NativeSymNodeImpl& /*n*/) { return false; })
      .def("maybe_as_int", &NativeSymNodeImpl::maybe_as_int)
      .def("str", &NativeSymNodeImpl::str)
      .def("statically_known_true", &NativeSymNodeImpl::statically_known_true)
      .def(
          "wrap_float",
          [](NativeSymNodeImpl& self, double v) {
            return node_to_py(self.wrap_float(v));
          })
      // Copies and pickles are Python SymNodes. _SymNode.__deepcopy__ clones.
      .def(
          "__deepcopy__",
          [](const NativeSymNodeImpl& self, py::handle memo) {
            return py::module_::import("copy").attr("deepcopy")(
                node_to_py(materialize(self)), memo);
          })
      .def(
          "__reduce_ex__",
          [](const NativeSymNodeImpl& self, py::handle /*protocol*/) {
            return py::make_tuple(
                py::module_::import("copy").attr("copy"),
                py::make_tuple(node_to_py(materialize(self))));
          })
      // Every other SymNode method runs its Python impl with the native node
      // as self; those read only the attributes bound here.
      .def(
          "__getattr__",
          [](py::handle self, const std::string& name) {
            py::handle cls = python_symnode_class();
            py::object attr = cls.attr("__dict__").attr("get")(name);
            bool dunder = name.size() > 4 && name.rfind("__", 0) == 0 &&
                name.compare(name.size() - 2, 2, "__") == 0;
            if (attr.is_none() || dunder) {
              // @allow-raw-throw: getattr and hasattr need an AttributeError
              throw py::attribute_error(
                  "'_NativeSymNode' object has no attribute '" + name + "'");
            }
            if (py::hasattr(attr, "__get__")) {
              return attr.attr("__get__")(self, cls);
            }
            return attr;
          })
      .def("__repr__", [](py::handle self) {
        return python_symnode_class().attr("__repr__")(self);
      });
  for (auto [name, fn] :
       {std::pair{"add", &c10::SymNodeImpl::add},
        std::pair{"sub", &c10::SymNodeImpl::sub},
        std::pair{"mul", &c10::SymNodeImpl::mul},
        std::pair{"truediv", &c10::SymNodeImpl::truediv},
        std::pair{"float_truediv", &c10::SymNodeImpl::float_truediv},
        std::pair{"int_truediv", &c10::SymNodeImpl::int_truediv},
        std::pair{"pow", &c10::SymNodeImpl::pow},
        std::pair{"float_pow", &c10::SymNodeImpl::float_pow},
        std::pair{"pow_by_natural", &c10::SymNodeImpl::pow_by_natural},
        std::pair{"floordiv", &c10::SymNodeImpl::floordiv},
        std::pair{"int_floordiv", &c10::SymNodeImpl::int_floordiv},
        std::pair{"mod", &c10::SymNodeImpl::mod},
        std::pair{"eq", &c10::SymNodeImpl::eq},
        std::pair{"ne", &c10::SymNodeImpl::ne},
        std::pair{"gt", &c10::SymNodeImpl::gt},
        std::pair{"lt", &c10::SymNodeImpl::lt},
        std::pair{"le", &c10::SymNodeImpl::le},
        std::pair{"ge", &c10::SymNodeImpl::ge},
        std::pair{"sym_min", &c10::SymNodeImpl::sym_min},
        std::pair{"sym_max", &c10::SymNodeImpl::sym_max},
        std::pair{"sym_and", &c10::SymNodeImpl::sym_and},
        std::pair{"sym_or", &c10::SymNodeImpl::sym_or},
        std::pair{"and_", &c10::SymNodeImpl::sym_and},
        std::pair{"or_", &c10::SymNodeImpl::sym_or}}) {
    node_cls.def(name, [fn](NativeSymNodeImpl& self, py::handle other) {
      return node_to_py((self.*fn)(node_from_py(other)));
    });
  }
  for (auto [name, fn] :
       {std::pair{"neg", &c10::SymNodeImpl::neg},
        std::pair{"sym_not", &c10::SymNodeImpl::sym_not},
        std::pair{"ceil", &c10::SymNodeImpl::ceil},
        std::pair{"floor", &c10::SymNodeImpl::floor},
        std::pair{"sym_float", &c10::SymNodeImpl::sym_float}}) {
    node_cls.def(name, [fn](NativeSymNodeImpl& self) {
      return node_to_py((self.*fn)());
    });
  }
  for (auto [name, fn] :
       {std::pair{"is_contiguous", &c10::SymNodeImpl::is_contiguous},
        std::pair{
            "is_channels_last_contiguous_2d",
            &c10::SymNodeImpl::is_channels_last_contiguous_2d},
        std::pair{
            "is_channels_last_contiguous_3d",
            &c10::SymNodeImpl::is_channels_last_contiguous_3d},
        std::pair{
            "is_channels_last_strides_2d",
            &c10::SymNodeImpl::is_channels_last_strides_2d},
        std::pair{
            "is_channels_last_strides_3d",
            &c10::SymNodeImpl::is_channels_last_strides_3d},
        std::pair{
            "is_non_overlapping_and_dense",
            &c10::SymNodeImpl::is_non_overlapping_and_dense}}) {
    node_cls.def(
        name,
        [fn](
            NativeSymNodeImpl& self,
            const py::sequence& sizes,
            const py::sequence& strides) {
          return node_to_py(
              (self.*fn)(nodes_from_py(sizes), nodes_from_py(strides)));
        });
  }
  node_cls.def(
      "is_non_overlapping_and_dense_indicator",
      [](NativeSymNodeImpl& self,
         const py::sequence& sizes,
         const py::sequence& strides) {
        return node_to_py(self.is_non_overlapping_and_dense_indicator(
            nodes_from_py(sizes), nodes_from_py(strides)));
      });
  node_cls.def(
      "sym_sum", [](NativeSymNodeImpl& self, const py::sequence& args) {
        return node_to_py(self.sym_sum(nodes_from_py(args)));
      });
  node_cls.def(
      "sym_ite",
      [](NativeSymNodeImpl& self, py::handle then_val, py::handle else_val) {
        return node_to_py(
            self.sym_ite(node_from_py(then_val), node_from_py(else_val)));
      });
}

} // namespace torch::symbolic
