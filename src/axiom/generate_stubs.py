# generate_stubs.py
from axiom import nn

def generate():
    content = '"""AUTO-GENERATED STUBS FOR IDE AUTOCOMPLETE. DO NOT EDIT."""\n'
    content += "from typing import Tuple, Any\n\n"

    # Grab all valid NN functions, explicitly ignoring loss functions!
    nn_funcs = {
        k: v for k, v in vars(nn).items()
        if callable(v)
           and not k.startswith('_')
           and not k.endswith('_loss')
           and not k.endswith('_logits')
    }

    # 1. Stubs for base Tensor
    content += "class NNTensorStubs:\n"
    for name in nn_funcs:
        content += f"    def {name}(self, *args, **kwargs) -> 'Any': ...\n"
    content += "\n"

    # 2. Stubs for TargetedTensor
    content += "class NNTargetedTensorStubs:\n"
    for name in nn_funcs:
        content += f"    def {name}(self, *args, **kwargs) -> 'Any': ...\n"
    content += "\n"

    # 3. Stubs for TargetedBundle
    content += "class NNTargetedBundleStubs:\n"
    for name in nn_funcs:
        content += f"    def {name}(self, *args, **kwargs) -> Tuple['Any', ...]: ...\n"

    with open("_nn_stubs.py", "w") as f:
        f.write(content)

    print("Ghost Mixins Generated! (src/axiom/_nn_stubs.py)")

if __name__ == "__main__":
    generate()