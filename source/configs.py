import torch

from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    center: bool = False

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


@dataclass
class ExperimentConfig:
    # optimization
    n_epochs: int = 200
    batch_size: int = 16
    lr: float = 2e-4
    sched_decay: float = 0.85
    betas: Tuple[float, float] = (0.8, 0.99)

    # training
    segment_length: int = 8192

    l1_gamma: float = 45.
    matching_gamma: float = 2.
    adv_gamma: float = 1.
    
    save_epochs: int = 2
    seed: int = 0xbebebe
    num_workers: int = 4
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # paths
    save_dir: str = 'checkpoints-V2'
    data_dir: str = 'data/LJSpeech-1.1/wavs'
    test_path: str = 'data/test/mels'
    eval_checkpoint = 'checkpoints/checkpoint_200ep.pth'

    # wandb
    project_name: str = 'hifigan'
    entity: str = 'i_vainn'

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)


def transform_dict(config_dict, expand = True):
    ret = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            t = transform_dict(dict(enumerate(v)), expand)
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret
