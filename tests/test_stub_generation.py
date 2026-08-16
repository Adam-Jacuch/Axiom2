import ast
from pathlib import Path

from axiom.generate_stubs import discover_axis_names, generate, render_stubs


def _class_body(source: str, class_name: str, next_class: str) -> str:
    return source.split(f"class {class_name}:", 1)[1].split(
        f"class {next_class}:", 1
    )[0]


def test_axis_discovery_finds_axiom_axes_without_matching_optax(tmp_path: Path):
    source_path = tmp_path / "model.py"
    source_path.write_text(
        "\n".join(
            [
                "from axiom import ax",
                "import optax",
                "feature = ax.feature",
                'token = ax("token", 32)',
                "optimizer = optax.adam(1e-3)",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "jax_only.py").write_text(
        "\n".join(
            [
                "import jax as ax",
                "key = ax.random.PRNGKey(0)",
            ]
        ),
        encoding="utf-8",
    )

    names = discover_axis_names([tmp_path])

    assert {"feature", "token"}.issubset(names)
    assert "adam" not in names
    assert "random" not in names


def test_rendered_stubs_are_typed_and_fluent(tmp_path: Path):
    source = render_stubs(scan_roots=[tmp_path], extra_axes=["custom_axis"])
    ast.parse(source)

    tensor_stubs = _class_body(
        source, "NNTensorStubs", "NNTargetedTensorStubs"
    )
    targeted_stubs = _class_body(
        source, "NNTargetedTensorStubs", "NNBundleStubs"
    )
    targeted_bundle_stubs = source.split(
        "class NNTargetedBundleStubs:", 1
    )[1]

    assert 'def custom_axis(self) -> "Axis"' in source
    assert 'def custom_axis(self) -> "TargetedTensor"' in tensor_stubs
    assert 'def gelu(self, /, approximate:' in tensor_stubs
    assert 'def gelu(self, /, approximate:' in targeted_stubs
    assert 'def softmax(self, /, where:' not in tensor_stubs
    assert 'def softmax(self, /, where:' in targeted_stubs
    assert 'def layer_norm(self, /, tie:' in targeted_stubs
    assert "seq_ax: 'Axis'" in targeted_stubs
    assert '-> "Tensor"' in targeted_stubs
    assert '-> "Bundle"' in targeted_bundle_stubs
    assert "ComplexWarning" not in source
    assert "def Tensor(" not in source


def test_generate_writes_only_to_the_requested_path(tmp_path: Path):
    output_path = tmp_path / "generated.py"

    result = generate(
        output_path=output_path,
        scan_roots=[tmp_path],
        extra_axes=["custom_axis"],
    )

    assert result == output_path
    assert output_path.exists()
    assert not (tmp_path / "_nn_stubs.py").exists()
    assert output_path.read_text(encoding="utf-8") == render_stubs(
        scan_roots=[tmp_path],
        extra_axes=["custom_axis"],
    )
