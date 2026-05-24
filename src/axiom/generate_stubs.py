import jax.numpy as jnp
import jax.nn as jnn
from axiom import nn

# These are explicit Python methods in core.py.
EXPLICIT_OVERRIDES = {
    'sum', 'mean', 'max', 'min', 'join', 'rename', 'unfold', 'pad', 'scan',
    'sample', 'mask', 'vmask', 'proj', 'bias', 'gate', 'pw', 'item', 'unwrap',
    'param', 'stop_grad', 'minimum', 'maximum', 'apply_n', 'tensor', 'bundle',
    'target_axes', 'merge', 'split'
}

# Block JAX physical array mutators from autocomplete
IGNORE_LIST = {
    'where', 'reshape', 'transpose', 'swapaxes', 'moveaxis', 'rollaxis',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'split',
    'array', 'asarray', 'empty', 'zeros', 'ones', 'full', 'empty_like',
    'zeros_like', 'ones_like', 'full_like', 'tile', 'repeat', 'choose',
    'take', 'put', 'insert', 'append', 'delete', 'diag', 'diagonal', 'squeeze',
    'expand_dims', 'argwhere', 'shape', 'ndim', 'size', 'copy', 'resize'
}


def get_callables(module):
    return {k: v for k, v in vars(module).items() if callable(v) and not k.startswith('_')}


def generate():
    content = '"""AUTO-GENERATED STUBS FOR IDE AUTOCOMPLETE. DO NOT EDIT."""\n'
    content += "from typing import Tuple, Any\n\n"

    jnp_funcs = get_callables(jnp)
    jnn_funcs = get_callables(jnn)

    axiom_nn_funcs = {
        k: v for k, v in get_callables(nn).items()
        if not k.endswith('_loss') and not k.endswith('_logits')
    }

    all_funcs = {**jnp_funcs, **jnn_funcs, **axiom_nn_funcs}

    # Filter out overrides AND the ignore list
    final_funcs = {
        k: v for k, v in all_funcs.items()
        if k not in EXPLICIT_OVERRIDES and k not in IGNORE_LIST
    }

    content += "class NNTensorStubs:\n"
    for name in sorted(final_funcs.keys()):
        content += f"    def {name}(self, *args, **kwargs) -> 'Any': ...\n"
    content += "\n"

    content += "class NNTargetedTensorStubs:\n"
    for name in sorted(final_funcs.keys()):
        content += f"    def {name}(self, *args, **kwargs) -> 'Any': ...\n"
    content += "\n"

    content += "class NNTargetedBundleStubs:\n"
    for name in sorted(final_funcs.keys()):
        content += f"    def {name}(self, *args, **kwargs) -> Tuple['Any', ...]: ...\n"

    with open("_nn_stubs.py", "w") as f:
        f.write(content)

    print(f"Ghost Mixins Generated! Mapped {len(final_funcs)} dynamic functions to IDE.")


if __name__ == "__main__":
    generate()