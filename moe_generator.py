import textwrap
import subprocess
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, PreTrainedModel
from transformers import Qwen2Tokenizer, Qwen2Model, Qwen2ForCausalLM, PretrainedConfig
import os
from pathlib import Path
import inspect
import transformers
transformers.logging.set_verbosity_error()


def get_function_parameters_list(func):
    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            if param.kind == param.VAR_POSITIONAL:
                params.append(f"*{name}")
            elif param.kind == param.VAR_KEYWORD:
                params.append(f"**{name}")
            else:
                params.append(name)
        else:
            params.append(f"{name}={param.default!r}")
    return params


def lower_case_model_name(name: str):
    new_name = name[0].lower()
    for c in name[1:]:
        if c.isupper():
            new_name += "_"
        new_name += c.lower()
    return new_name

def pascal_case_model_name(name: str):
    return ''.join(list(map(lambda n: n[0].upper() + n[1:], name.split('_'))))


def generate_moe_modeling(
        model_name: str,
        moe_model_name: str,
        move_to_cwd=False):

    model = AutoModel.from_pretrained(model_name)

    moe_class_name = moe_model_name

    base_model = model
    base_model_name = base_model.__class__.__name__[:-5]
    base_model_class: type[PreTrainedModel] = base_model.__class__

    base_model_config_class = base_model.config.__class__
    modeling_path = base_model_class.__module__
    config_path = base_model_config_class.__module__
    base_config_params_string = ",\n".join(get_function_parameters_list(base_model_config_class.__init__)[1:-1])

    modular_string = f"""
from typing import Callable, Optional, Tuple

import random

import torch
import torch.utils.checkpoint
from torch import nn

from {modeling_path} import (
    {base_model_name}Model,
    {base_model_name}MLP,
    {base_model_name}DecoderLayer,
    {base_model_name}ForCausalLM,
    {base_model_name}PreTrainedModel
)

from {config_path} import (
    {base_model_name}Config
)


from transformers.utils import logging


logger = logging.get_logger(__name__)

class {moe_class_name}Config({base_model_name}Config):
    def __init__(
        self,
{textwrap.indent(base_config_params_string, "        ")},
        num_experts=3,
        num_experts_per_tok=2,
        moe_layers_idx=(0,),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layers_idx = tuple(moe_layers_idx)


class {moe_class_name}MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok= config.num_experts_per_tok
        self.config = config
        self.experts = nn.ModuleList([{moe_class_name}MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        chosen_experts_idx = random.sample(list(range(len(self.experts))), k=self.num_experts_per_tok)

        s = self.experts[chosen_experts_idx[0]].forward(x)
        for expert_idx in chosen_experts_idx[1:]:
            s_new = self.experts[expert_idx].forward(x)
            s += s_new
        s /= self.num_experts_per_tok
        return s


class {moe_class_name}DecoderLayer({base_model_name}DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = {moe_class_name}MoEBlock(config) if layer_idx in config.moe_layers_idx else {moe_class_name}MLP(config)

class {moe_class_name}MLP({base_model_name}MLP):
    pass

class {moe_class_name}PreTrainedModel({base_model_name}PreTrainedModel):
    pass

class {moe_class_name}ForCausalLM({base_model_name}ForCausalLM):
    pass

class {moe_class_name}Model({base_model_name}Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [{moe_class_name}DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


__all__ = [
    "{moe_class_name}Model",
    "{moe_class_name}PreTrainedModel",
    "{moe_class_name}ForCausalLM",
    "{moe_class_name}Config",
]
"""

    low_model_name = lower_case_model_name(moe_model_name)
    print(f"Generating MOE model {moe_class_name} from base model {base_model_name}...")
    moe_model_dir = f"transformers/src/transformers/models/{low_model_name}"
    os.makedirs(moe_model_dir, exist_ok=True)
    with open(f"{moe_model_dir}/modular_{low_model_name}.py", "w") as modular_file:
        modular_file.write(modular_string)

    # generate modeling and configuration files
    import sys
    out = subprocess.run(
        [sys.executable, 'utils/modular_model_converter.py', '--files_to_parse',
         f'{os.getcwd()}/{moe_model_dir}/modular_{low_model_name}.py'],
        env=os.environ.copy(),
        cwd="transformers/",
        capture_output=True,
        text=True
    )

    print(textwrap.indent(out.stdout + out.stderr, "    "), end="")

    model_init_file = f"""
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_{low_model_name} import *
    from .modeling_{low_model_name} import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
"""
    with open(f"{moe_model_dir}/__init__.py", "w") as f:
        f.write(model_init_file)

    if not move_to_cwd:
        print("Generation finished!")
        return 'transformers/models', low_model_name
    # move model to local path and replace relative imports with normal ones
    print("    Moving model to local folder...")
    import shutil
    if os.path.exists(low_model_name):
        shutil.rmtree(low_model_name)
    shutil.move(moe_model_dir, ".")

    with open(f"{low_model_name}/__init__.py", "w") as f:
        f.write(model_init_file)

    for filename in [f"modeling_{low_model_name}.py", f"configuration_{low_model_name}.py", "__init__.py"]:
        with open(f"{low_model_name}/{filename}") as f:
            modeling = f.read()
        with open(f"{low_model_name}/{filename}", "w") as f:
            f.write(modeling.replace("from ...", "from transformers.").replace("from .", f"from {low_model_name}."))
    print("Generation finished!")

    return '', low_model_name


def load_weights_to_moe(moe_model, base_model):
    if hasattr(base_model, "model"):
        base_model = base_model.model
    if hasattr(moe_model, "model"):
        moe_model = moe_model.model
    for layer_idx in moe_model.config.moe_layers_idx:
        print(f"Copying layer {layer_idx}...")
        for expert_idx in range(moe_model.config.num_experts):
            print(f"   copying to expert {expert_idx}...", end='')
            with torch.no_grad():
                moe_model.layers[layer_idx].mlp.experts[expert_idx].load_state_dict(
                    base_model.layers[layer_idx].mlp.state_dict())
            print("done!")
    print("Copying finished!")


def init_and_save_custom_moe_weights(moe_class: type,
                                     base_pretrained_model_name_or_path: str,
                                     moe_target_name_or_path: str):
    print("Initializing base and moe models...")
    base_model = AutoModelForCausalLM.from_pretrained(base_pretrained_model_name_or_path)
    moe_model = moe_class.from_pretrained(base_pretrained_model_name_or_path)
    load_weights_to_moe(moe_model, base_model)
    print("Saving moe...")
    moe_model.save_pretrained(moe_target_name_or_path, safe_serialization=True)
    print("Saving tokenizer...")
    AutoTokenizer.from_pretrained(base_pretrained_model_name_or_path).save_pretrained(moe_target_name_or_path)
    print(f"Saved {moe_target_name_or_path}!")


def load_weights_to_moe2(moe_model, base_models: list[(str, PreTrainedModel)],):

    for layer_idx in moe_model.config.moe_layers_idx:
        print(f"Copying layer {layer_idx}...")
        for expert_idx in range(moe_model.config.num_experts):
            print(f"   copying to expert {expert_idx}...", end='')
            with torch.no_grad():
                moe_model.layers[layer_idx].mlp.experts[expert_idx].load_state_dict(
                    base_models[expert_idx].layers[layer_idx].mlp.state_dict())
            print("done!")
    print("Copying finished!")


def load_donored_mlps_to_experts(moe_modeling_folder: str,
                                 attention_donor_path: str,
                                 mlp_donors_paths: list[str],
                                 moe_target_name_or_path: str,
                                 num_experts_per_tok: int,
                                 moe_layers_idx: list[int]):

    if num_experts_per_tok > len(mlp_donors_paths):
        print(f"Error: num_experts_per_tok > num_experts (length of mlp_donors_paths).")
        return


    import importlib
    from pathlib import Path
    moe_modeling_folder = Path(moe_modeling_folder)
    model_name = pascal_case_model_name(moe_modeling_folder.name)
    modeling_path = str(moe_modeling_folder / f"modeling_{moe_modeling_folder.name}").replace('/', '.')
    config_path = str(moe_modeling_folder / f"configuration_{moe_modeling_folder.name}").replace('/', '.')
    moe_model_module = importlib.import_module(modeling_path)
    moe_model_class: type[PreTrainedModel] = getattr(moe_model_module, f"{model_name}ForCausalLM")
    moe_config_module = importlib.import_module(config_path)
    moe_config_class: type[PretrainedConfig] = getattr(moe_config_module, f"{model_name}Config")

    print("Initializing base and moe models...")
    config = moe_config_class.from_pretrained(attention_donor_path)
    config.update({
        "num_experts": len(mlp_donors_paths),
        "num_experts_per_tok": num_experts_per_tok,
        "moe_layers_idx": moe_layers_idx}
    )
    moe_model = moe_model_class.from_pretrained(attention_donor_path, config=config)
    base_models_map = dict()
    base_models = []
    for m in mlp_donors_paths:
        if m not in base_models_map:
            base_models_map[m] = AutoModelForCausalLM.from_pretrained(m)
        base_models.append(base_models_map[m])

    for layer_idx in moe_model.config.moe_layers_idx:
        print(f"    Loading layer {layer_idx}...")
        for expert_idx in range(moe_model.config.num_experts):
            print(f"       copying {mlp_donors_paths[expert_idx]} mlp to expert {expert_idx}...", end='')
            with torch.no_grad():
                moe_model.model.layers[layer_idx].mlp.experts[expert_idx].load_state_dict(
                    base_models[expert_idx].model.layers[layer_idx].mlp.state_dict())
            print("done!")
    print("Copying finished!")

    print("Saving moe...")
    moe_model.save_pretrained(moe_target_name_or_path, safe_serialization=True)
    print("Saving tokenizer...")
    AutoTokenizer.from_pretrained(attention_donor_path).save_pretrained(moe_target_name_or_path)
    print(f"Model config and weights saved to {moe_target_name_or_path}!")

def run_generator(
    moe_name,
    attention_donor,
    mlp_donors,
    num_experts_per_tok,
    moe_layers_idx,
    move_to_cwd
):
    modeling_parent_path, model_name = generate_moe_modeling(attention_donor, pascal_case_model_name(moe_name), move_to_cwd=move_to_cwd)
    print()
    load_donored_mlps_to_experts(
        str(Path(modeling_parent_path) / model_name),
        attention_donor,
        mlp_donors,
        moe_name + f"-{len(mlp_donors)}exp",
        num_experts_per_tok=num_experts_per_tok,
        moe_layers_idx=moe_layers_idx
    )

def compare_models():



if __name__ == '__main__':
    import sys, yaml
    with open(sys.argv[1]) as stream:
        params = yaml.safe_load(stream)
        run_generator(
            params['name'],
            params['attention'],
            params['experts'],
            params['num_experts_per_tok'],
            params['moe_layers_idx'],
            ("--move_to_cwd" in sys.argv)
        )

"""
cd transformers
pip install -e .
pip install torch
pip install libcst
pip install ruff
"""
