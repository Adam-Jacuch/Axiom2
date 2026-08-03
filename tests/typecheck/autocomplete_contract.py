from typing_extensions import assert_type

from axiom import ax
from axiom.core import Axis, Bundle, TargetedBundle, TargetedTensor, Tensor


def check_tensor_autocomplete(x: Tensor) -> None:
    assert_type(ax.d, Axis)
    assert_type(ax.custom_axis, Axis)
    assert_type(x.d, TargetedTensor)
    assert_type(x.d.s, TargetedTensor)

    assert_type(x.clip(0, None), Tensor)
    assert_type(x.d.gelu(approximate=False), Tensor)
    assert_type(x.d.softmax(where=None), Tensor)
    assert_type(x.d.layer_norm(eps=1e-5), Tensor)
    assert_type(x.d.rope(seq_ax=ax.s), Tensor)
    assert_type(x.d.proj(ax.d(8)).d.bias(), Tensor)


def check_bundle_autocomplete(bundle: Bundle) -> None:
    assert_type(bundle.d, TargetedBundle)
    assert_type(bundle.d.s, TargetedBundle)
    assert_type(bundle.d.silu(), Bundle)
    assert_type(bundle.d.proj(ax.h(8)).h.bias(), Bundle)
