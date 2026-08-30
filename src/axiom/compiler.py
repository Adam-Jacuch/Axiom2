import jax
from jax.tree_util import register_pytree_node
from functools import wraps
import inspect
from typing import Callable, Any, TypeVar, Optional, Tuple, Dict
from .core import compiler_state, decay_monads
from .layout import AxiomLayout, AxiomMesh

F = TypeVar('F', bound=Callable[..., Any])

# Ensure the tracer routing attributes exist on the singleton
compiler_state.step_params = None


# ==========================================
# 1. THE PYTREE MODEL
# ==========================================
class AxiomModel:
    """A mathematically pure PyTree wrapper for Axiom functions."""

    def __init__(
        self,
        fn: Callable,
        params: Optional[Dict] = None,
        *,
        mesh: Optional[AxiomMesh] = None,
        param_layouts: Optional[Dict] = None,
    ):
        self.fn = fn
        self.is_initialized = params is not None
        self.params = params if params is not None else {}
        self.mesh = mesh
        self.param_layouts = dict(param_layouts or {})

    @property
    def layout(self) -> AxiomLayout:
        if self.mesh is None:
            raise ValueError("This model has no device mesh. Construct it with ax.model(fn, mesh=mesh).")
        return AxiomLayout(self.mesh, self.param_layouts)

    def __call__(self, *args, **kwargs):
        args = decay_monads(args)
        kwargs = decay_monads(kwargs)

        # 1. Snapshot the current trace state
        prev_params = getattr(compiler_state, 'params', {})
        prev_layouts = getattr(compiler_state, 'param_layouts', {})
        prev_counter = getattr(compiler_state, 'param_counter', 0)
        prev_frames = compiler_state.active_frames.copy()
        prev_calls = compiler_state.func_calls.copy()
        prev_initializing = compiler_state.is_initializing
        prev_remat_scope = compiler_state.remat_scope_override
        prev_strict_params = compiler_state.strict_params

        compiler_state.params = self.params
        compiler_state.param_layouts = self.param_layouts
        compiler_state.param_counter = 0
        compiler_state.active_frames.clear()
        compiler_state.func_calls.clear()

        try:
            is_uninitialized = not self.is_initialized
            compiler_state.strict_params = not is_uninitialized
            if prev_initializing or is_uninitialized:
                compiler_state.is_initializing = True
                res = self.fn(*args, **kwargs)
                self.params = compiler_state.params.copy()
                self.param_layouts = compiler_state.param_layouts.copy()
                self.is_initialized = True
            else:
                res = self.fn(*args, **kwargs)
            return res
        finally:
            compiler_state.is_initializing = prev_initializing
            compiler_state.param_counter = prev_counter
            compiler_state.active_frames = prev_frames
            compiler_state.func_calls = prev_calls
            compiler_state.remat_scope_override = prev_remat_scope
            compiler_state.strict_params = prev_strict_params
            compiler_state.params = prev_params
            compiler_state.param_layouts = prev_layouts

    def init(self, *topology: 'Axis', **kwargs) -> 'AxiomModel':
        import jax.numpy as jnp
        from .core import Tensor

        for a in topology:
            if not hasattr(a, 'size') or a.size is None:
                raise ValueError(f"AxiomModel.init() requires strictly sized Axes, got: {a}")

        shape = tuple(a.size for a in topology)
        dummy_input = Tensor(jnp.zeros(shape, dtype=jnp.int32), *topology)
        # Allocation sites mutate the backing dictionaries during the eager
        # initialization pass, so retain shallow snapshots rather than aliases.
        previous_params = self.params.copy()
        previous_layouts = self.param_layouts.copy()
        previous_initialized = self.is_initialized
        try:
            _ = self(dummy_input, **kwargs)
            if self.mesh is not None:
                self.params = self.layout.place_params(self.params)
        except Exception:
            # A failed placement (for example h[tp](5) on tp=2) must not
            # leave a model with partially allocated, unusable parameters.
            self.params = previous_params
            self.param_layouts = previous_layouts
            self.is_initialized = previous_initialized
            raise

        return self

    def astype(self, dtype) -> 'AxiomModel':
        import jax
        self.params = jax.tree.map(lambda p: p.astype(dtype), self.params)
        return self

    def _with_params(self, params: Dict) -> 'AxiomModel':
        return AxiomModel(self.fn, params, mesh=self.mesh, param_layouts=self.param_layouts)

    def __sub__(self, other):
        if isinstance(other, dict):
            new_params = jax.tree_util.tree_map(lambda p, g: p - g, self.params, other)
            return self._with_params(new_params)
        raise TypeError("Can only subtract parameter dictionaries from an AxiomModel.")

    def __add__(self, other):
        if isinstance(other, AxiomModel):
            new_params = jax.tree_util.tree_map(lambda p1, p2: p1 + p2, self.params, other.params)
            return self._with_params(new_params)
        if isinstance(other, dict):
            new_params = jax.tree_util.tree_map(lambda p, g: p + g, self.params, other)
            return self._with_params(new_params)
        raise TypeError("Can only add parameter dicts or AxiomModels to an AxiomModel.")

    def __mul__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            new_params = jax.tree_util.tree_map(lambda p: p * scalar, self.params)
            return self._with_params(new_params)
        raise TypeError("Can only multiply an AxiomModel by a scalar float/int.")

    def items(self): return self.params.items()
    def keys(self): return self.params.keys()
    def values(self): return self.params.values()
    def __getitem__(self, key: str): return self.params[key]
    def __setitem__(self, key: str, value: Any): self.params[key] = value
    def __contains__(self, key: str): return key in self.params
    def __iter__(self): return iter(self.params)
    def __len__(self): return len(self.params)


def _layout_aux(layouts):
    def axis_data(axis):
        return axis.name, axis.size, axis.placement, axis.replicated

    return tuple(
        (
            name,
            layout.kind,
            tuple(axis_data(axis) for axis in layout.axes),
            tuple(axis_data(axis) for axis in layout.input_axes),
            tuple(axis_data(axis) for axis in layout.output_axes),
        )
        for name, layout in sorted(layouts.items())
    )


def _layouts_from_aux(layout_items):
    from .core import Axis
    from .layout import ParameterLayout

    def axis_from_data(data):
        name, size, placement, replicated = data
        return Axis(name, size, placement, replicated=replicated)

    return {
        name: ParameterLayout(
            axes=tuple(axis_from_data(axis) for axis in axes),
            kind=kind,
            input_axes=tuple(axis_from_data(axis) for axis in input_axes),
            output_axes=tuple(axis_from_data(axis) for axis in output_axes),
        )
        for name, kind, axes, input_axes, output_axes in layout_items
    }


def _unflatten_model(aux, children):
    fn, mesh, layout_items = aux
    params = children[0]
    model = AxiomModel(fn, params, mesh=mesh, param_layouts=_layouts_from_aux(layout_items))
    model.is_initialized = True
    return model

register_pytree_node(
    AxiomModel,
    lambda m: ((m.params,), (m.fn, m.mesh, _layout_aux(m.param_layouts))),
    _unflatten_model
)


# ==========================================
# 2. GHOST PASS & SHARDING WRAPPERS
# ==========================================
def _trigger_ghost_pass(fn, *args, **kwargs):
    def _get_models(obj):
        models = []
        if isinstance(obj, AxiomModel):
            models.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                models.extend(_get_models(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                models.extend(_get_models(item))
        return models

    models = _get_models(args) + _get_models(kwargs)

    if any(not m.is_initialized for m in models):
        previous_initializing = compiler_state.is_initializing
        previous_remat_scope = compiler_state.remat_scope_override
        compiler_state.is_initializing = True
        compiler_state.reset_pass_state()
        try:
            fn(*args, **kwargs)
        finally:
            compiler_state.is_initializing = previous_initializing
            compiler_state.remat_scope_override = previous_remat_scope


class AxiomJitWrapper:
    def __init__(self, fn, static_argnames=None, mesh: Optional[AxiomMesh] = None):
        self.fn = fn
        self.static_argnames = static_argnames
        self.mesh = mesh
        # One executable cannot safely own two different explicit layout
        # trees.  Keep a compact cache per input placement/static signature;
        # regular JAX still owns shape/dtype specialization inside each entry.
        self._jitted_fns = {}

    def __call__(self, *args, **kwargs):
        # A sliced monad is a temporary editing view; compiled functions operate
        # on its concrete chunk tensor just as model calls and exports do.
        args = decay_monads(args)
        kwargs = decay_monads(kwargs)
        # pjit accepts explicit in_shardings only for positional arguments.
        # Normalize ordinary Python keyword calls up front, preserving static
        # argument names for the later JAX transform.
        static_argnums = None
        if self.mesh is not None:
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            bound.apply_defaults()
            args, kwargs = bound.args, bound.kwargs
            if kwargs:
                raise TypeError(
                    "@ax.jit(mesh=...) does not support keyword-only dynamic arguments with explicit layouts; "
                    "pass them positionally or close over them."
                )
            static_names = (self.static_argnames,) if isinstance(self.static_argnames, str) else tuple(self.static_argnames or ())
            positions = {
                parameter.name: index
                for index, parameter in enumerate(inspect.signature(self.fn).parameters.values())
                if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            }
            missing = [name for name in static_names if name not in positions]
            if missing:
                raise TypeError(
                    "@ax.jit(mesh=...) static_argnames must name positional parameters; unsupported: "
                    f"{missing}."
                )
            static_argnums = tuple(positions[name] for name in static_names) or None

        # 1. Trigger the Ghost Pass if uninitialized
        _trigger_ghost_pass(self.fn, *args, **kwargs)

        # 2. Lazy compilation.  Sharding comes from each Tensor's named axes,
        # never from a positional list supplied to the decorator.
        if self.mesh is not None:
            compilation_key = _mesh_compilation_key(args, static_argnums, self.mesh)
            jitted_fn = self._jitted_fns.get(compilation_key)
            if jitted_fn is None:
                static_positions = set(static_argnums or ())
                static_values = {index: args[index] for index in static_positions}
                dynamic_args = tuple(arg for index, arg in enumerate(args) if index not in static_positions)

                def rebuild_args(dynamic_values):
                    dynamic_iter = iter(dynamic_values)
                    return tuple(
                        static_values[index] if index in static_positions else next(dynamic_iter)
                        for index in range(len(args))
                    )

                in_shardings = tuple(
                    _argument_sharding(arg, self.mesh)
                    for index, arg in enumerate(args)
                    if index not in static_positions
                )
                # Trace only abstract values to recover the output Tensor
                # topology before compiling.  This gives JAX an explicit
                # out_shardings tree even when a kernel body hides dataflow
                # from GSPMD's usual propagation heuristics.
                def call_with_static(*f_dynamic_args):
                    return self.fn(*rebuild_args(f_dynamic_args))

                with self.mesh.jax_mesh:
                    abstract_output = jax.eval_shape(call_with_static, *dynamic_args)
                out_shardings = _argument_sharding(abstract_output, self.mesh)

                def mesh_wrapped_fn(*f_args, **f_kwargs):
                    with self.mesh.jax_mesh:
                        return call_with_static(*f_args)

                jitted_fn = jax.jit(
                    mesh_wrapped_fn,
                    in_shardings=in_shardings,
                    out_shardings=out_shardings,
                )
                self._jitted_fns[compilation_key] = jitted_fn
            dynamic_args = tuple(
                arg for index, arg in enumerate(args)
                if static_argnums is None or index not in static_argnums
            )
            return jitted_fn(*dynamic_args)

        # Unmeshed compilation keeps JAX's normal polymorphic cache behavior.
        jitted_fn = self._jitted_fns.get(None)
        if jitted_fn is None:
            jitted_fn = jax.jit(self.fn, static_argnames=self.static_argnames)
            self._jitted_fns[None] = jitted_fn
        return jitted_fn(*args, **kwargs)


class AxiomGradWrapper:
    def __init__(self, fn, has_aux=False, is_value_and_grad=False):
        self.fn = fn
        self.has_aux = has_aux
        self.is_value_and_grad = is_value_and_grad

    def __call__(self, *args, **kwargs):
        import jax.numpy as jnp
        import jax

        if getattr(compiler_state, 'is_initializing', False):
            out = self.fn(*args, **kwargs)
            m = args[0]
            mock_params = {k: jnp.zeros_like(v.unwrap() if hasattr(v, 'unwrap') else v) for k, v in m.params.items()}
            mock_grads = AxiomModel(m.fn, mock_params, mesh=m.mesh, param_layouts=m.param_layouts)

            if self.is_value_and_grad:
                return out, mock_grads
            return mock_grads

        def jax_compatible_fn(*f_args, **f_kwargs):
            out = self.fn(*f_args, **f_kwargs)
            if self.has_aux:
                loss_tensor, aux = out
                return jnp.sum(loss_tensor.unwrap()), out
            else:
                loss_tensor = out
                return jnp.sum(loss_tensor.unwrap()), out

        if self.is_value_and_grad:
            (_, smuggled_out), grads = jax.value_and_grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            return smuggled_out, grads
        else:
            grads, smuggled_out = jax.grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            if self.has_aux:
                return grads, smuggled_out[1]
            return grads


# ==========================================
# 3. THE PUBLIC API
# ==========================================

def _argument_sharding(value: Any, mesh: AxiomMesh):
    """Mirror an Axiom value's PyTree while replacing arrays with NamedSharding."""
    from .core import Bundle, Tensor

    if isinstance(value, Tensor):
        return mesh.named_sharding(AxiomLayout(mesh, {}).tensor_spec(value))
    if isinstance(value, AxiomModel):
        if value.mesh is not None and value.mesh is not mesh:
            raise ValueError("A model and @ax.jit(mesh=...) must use the same ax.mesh() instance.")
        layout = AxiomLayout(mesh, value.param_layouts)
        return AxiomModel(value.fn, layout.parameter_shardings(value.params), mesh=mesh,
                          param_layouts=value.param_layouts)
    if isinstance(value, Bundle):
        return Bundle(*[_argument_sharding(tensor, mesh) for tensor in value.tensors])
    if isinstance(value, tuple):
        return tuple(_argument_sharding(item, mesh) for item in value)
    if isinstance(value, list):
        return [_argument_sharding(item, mesh) for item in value]
    if isinstance(value, dict):
        return {key: _argument_sharding(item, mesh) for key, item in value.items()}
    return None


def _mesh_compilation_key(args: tuple[Any, ...], static_argnums: tuple[int, ...] | None, mesh: AxiomMesh):
    """Return a hashable key for layouts fixed into a mesh ``jax.jit`` call."""
    dynamic_args = tuple(
        arg for index, arg in enumerate(args)
        if static_argnums is None or index not in static_argnums
    )
    sharding_tree = _argument_sharding(dynamic_args, mesh)
    leaves, tree = jax.tree_util.tree_flatten(sharding_tree)

    def leaf_signature(leaf):
        spec = getattr(leaf, "spec", None)
        if spec is not None:
            return ("named", tuple(spec))
        return (type(leaf).__qualname__, repr(leaf))

    static_values = tuple(args[index] for index in (static_argnums or ()))
    return (
        repr(tree),
        tuple(leaf_signature(leaf) for leaf in leaves),
        tuple(_static_argument_signature(value) for value in static_values),
    )


def _static_argument_signature(value: Any):
    """Hash static arguments by all metadata relevant to Axiom compilation."""
    from .core import Axis

    # Axis equality intentionally compares logical names only for tensor
    # algebra.  A compilation cache must also distinguish size and placement:
    # ``ax.h[tp]`` and ``ax.h[None]`` need different executables.
    if isinstance(value, Axis):
        placement = getattr(value, "placement", None)
        return (
            "axis",
            value.name,
            value.size,
            getattr(placement, "mesh_id", None),
            getattr(placement, "name", None),
            value.replicated,
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_static_argument_signature(item) for item in value))
    if isinstance(value, frozenset):
        return ("frozenset", frozenset(_static_argument_signature(item) for item in value))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("@ax.jit(mesh=...) static arguments must be hashable.") from exc
    return type(value).__module__, type(value).__qualname__, value


def jit(fn=None, *, static_argnames=None, mesh: Optional[AxiomMesh] = None, shard=None):
    """Compile an Axiom function; axis placement is declared with ``x.d[mesh.tp]``."""
    if shard is not None:
        raise TypeError(
            "ax.jit(shard=...) has been removed. Create mesh = ax.mesh(...) and annotate logical axes, "
            "then use @ax.jit(mesh=mesh)."
        )
    if mesh is not None and not isinstance(mesh, AxiomMesh):
        raise TypeError("mesh= must be the value returned by ax.mesh(...).")
    if fn is None:
        return lambda f: AxiomJitWrapper(f, static_argnames=static_argnames, mesh=mesh)
    return AxiomJitWrapper(fn, static_argnames=static_argnames, mesh=mesh)


def value_and_grad(fn=None, has_aux=False):
    if fn is None:
        return lambda f: AxiomGradWrapper(f, has_aux=has_aux, is_value_and_grad=True)
    return AxiomGradWrapper(fn, has_aux=has_aux, is_value_and_grad=True)


def grad(fn=None, has_aux=False):
    if fn is None:
        return lambda f: AxiomGradWrapper(f, has_aux=has_aux, is_value_and_grad=False)
    return AxiomGradWrapper(fn, has_aux=has_aux, is_value_and_grad=False)


def apply_updates(model: AxiomModel, grads: Any, optimizer: Any, opt_state: Any) -> Tuple[AxiomModel, Any]:
    import optax
    if isinstance(grads, dict):
        grads = AxiomModel(model.fn, grads, mesh=model.mesh, param_layouts=model.param_layouts)
    if opt_state is None:
        opt_state = optimizer.init(model)
        if model.mesh is not None:
            # Optax's usual zeros_like state already inherits placement, but
            # normalizing it here also covers transformations that allocate
            # host arrays or counters during init.
            opt_state = model.layout.place_state(opt_state, model.params)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = optax.apply_updates(model, updates)
    if model.mesh is not None:
        new_model.params = model.layout.place_params(new_model.params)
        new_opt_state = model.layout.place_state(new_opt_state, new_model.params)
    return new_model, new_opt_state


def to_jax(model, *init_axes: 'Axis', mesh: Optional[AxiomMesh] = None, sharding: bool = False, **kwargs):
    """Export Axiom parameters and an apply function for native JAX tooling.

    ``ax.to_jax(model)`` retains the historical two-value return.  Supplying
    ``sharding=True`` returns ``(params, apply_fn, layout)`` where ``layout``
    exposes concrete ``PartitionSpec``/``NamedSharding`` trees and optimizer
    state helpers.
    """
    import inspect
    if inspect.isfunction(model):
        from axiom import ax
        model = ax.model(model)

    if init_axes and not model.is_initialized:
        import jax.numpy as jnp
        from .core import Tensor, Axis
        shape = []
        for a in init_axes:
            if not isinstance(a, Axis) or a.size is None:
                raise ValueError(f"Auto-initialization requires strictly sized Axes, got: {a}")
            shape.append(a.size)
        dummy_input = Tensor(jnp.zeros(shape), *init_axes)
        _ = model(dummy_input, **kwargs)

    if not model.is_initialized:
        raise ValueError("AxiomModel must be initialized.")

    effective_mesh = mesh if mesh is not None else model.mesh
    if mesh is not None and model.mesh is not None and mesh is not model.mesh:
        raise ValueError("to_jax(mesh=...) must use the same mesh that owns the model's axis annotations.")
    if sharding and effective_mesh is None:
        raise ValueError("to_jax(..., sharding=True) requires mesh=... or a model created with mesh=....")
    layout = AxiomLayout(effective_mesh, model.param_layouts) if effective_mesh is not None else None
    params = model.params.copy()
    if sharding:
        params = layout.place_params(params)

    def apply_fn(params_dict, *args, **apply_kwargs):
        from .core import decay_monads, compiler_state
        args = decay_monads(args)
        apply_kwargs = decay_monads(apply_kwargs)

        prev_params = getattr(compiler_state, 'params', {})
        prev_layouts = getattr(compiler_state, 'param_layouts', {})
        prev_counter = getattr(compiler_state, 'param_counter', 0)
        prev_frames = compiler_state.active_frames.copy()
        prev_calls = compiler_state.func_calls.copy()
        prev_remat_scope = compiler_state.remat_scope_override
        prev_strict_params = compiler_state.strict_params

        compiler_state.params = params_dict
        compiler_state.param_layouts = model.param_layouts
        compiler_state.strict_params = True
        compiler_state.param_counter = 0
        compiler_state.active_frames.clear()
        compiler_state.func_calls.clear()

        try:
            res = model.fn(*args, **apply_kwargs)
        finally:
            compiler_state.param_counter = prev_counter
            compiler_state.active_frames = prev_frames
            compiler_state.func_calls = prev_calls
            compiler_state.remat_scope_override = prev_remat_scope
            compiler_state.strict_params = prev_strict_params
            compiler_state.params = prev_params
            compiler_state.param_layouts = prev_layouts

        return res

    return (params, apply_fn, layout) if sharding else (params, apply_fn)
