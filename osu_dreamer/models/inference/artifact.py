
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type, TypeVar

import torch as th

from .model import LDM, LDMArgs

def save_inference(
    latent_ckpt_path: str,
    denoiser_ckpt_path: str,
    style_prior_ckpt_path: str,
    output_path: str,
):
    latent_ckpt = th.load(latent_ckpt_path, map_location='cpu')
    denoiser_ckpt = th.load(denoiser_ckpt_path, map_location='cpu')
    style_prior_ckpt = th.load(style_prior_ckpt_path, map_location='cpu')
    inference_hparams = {
        **{ k: latent_ckpt['hyper_parameters'][k] for k in ['emb_dim', 'style_dim', 'n_downs', 'stride', 'latent_args'] },
        **{ k: denoiser_ckpt['hyper_parameters'][k] for k in ['diffusion_args'] },
        **{ k: style_prior_ckpt['hyper_parameters'][k] for k in ['style_prior_args'] },
    }

    inference_state_dict = {
        **{ k: v for k, v in latent_ckpt['state_dict'].items() if k.startswith('latent.') },
        **{ k: v for k, v in denoiser_ckpt['state_dict'].items() if k.startswith('diffusion.') },
        **{ k: v for k, v in style_prior_ckpt['state_dict'].items() if k.startswith('style_prior.') },
    }

    with open(output_path, 'wb') as f:
        th.save({
            'hparams': inference_hparams,
            'state_dict': inference_state_dict,
        }, f)

def load_inference(model_path) -> LDM:
    model_artifact = th.load(model_path)
    model = LDM(dataclass_from_dict(LDMArgs, model_artifact['hparams']))
    model.load_state_dict(model_artifact['state_dict'])
    model = model.eval()
    return model

T = TypeVar("T")
def dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs = {}
    field_types = {f.name: f.type for f in fields(cls)}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Recursively handle nested dataclasses
        if is_dataclass(field_type) and isinstance(field_type, type) and isinstance(value, dict):
            kwargs[key] = dataclass_from_dict(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)