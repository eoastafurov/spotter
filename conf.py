from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class Model:
    kernels: Tuple[int]
    strides: Tuple[int]
    channels: Tuple[int]
    hidden_size: int
    activation: str


@dataclass
class Optim:
    lr: float
    n_epochs: int
    batch_size: int
        

@dataclass
class Features:
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int


@dataclass
class Augmentations:
    freq_mask_param: int
    time_mask_param: int


@dataclass
class ExpConfig:
    sample_rate: int = 16_000
    val_fraction: float = 0.1
    idx_to_keyword: List[str] = ('sber', 'joy', 'afina', 'salut', 'filler')
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model: Model = Model(
        # kernels=(3, 3), strides=(2, 2), channels=(32, 32), hidden_size=16, activation='ReLU'
        # kernels=(6, 6, 6, 6, 6, 6, 6), strides=(1, 1, 1, 1, 1, 1, 1, 1), channels=(32, 32, 32, 32, 32, 32, 32, 32), hidden_size=16, activation='SiLU'
        kernels=(6, 6, 6, 6, 6, 6, 3, 3, 3, 3), strides=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), channels=(32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16), hidden_size=16, activation='SiLU'

    )
    optim: Optim = Optim(
        lr=1e-3, n_epochs=10, batch_size=2048
    )
    features: Features = Features(
        n_fft=400, win_length=400, hop_length=160, n_mels=64
    )
    augs: Augmentations = Augmentations(
        freq_mask_param=15, time_mask_param=15
    )


@dataclass
class PretrainConfig:
    # experimantal: Experimantal(
    #     pretrain=True
    # )
    sample_rate: int = 16_000
    val_fraction: float = 0.1
    idx_to_keyword: List[str] = ('on', 'sheila', 'cat', 'left', 'bird', 'no', 'dog', 'seven', 'forward', 'two', 'backward', 'yes', 'off', 'up', 'happy', 'wow', 'house', 'three', 'stop', 'marvin', 'tree', 'bed', 'five', 'right', 'eight', 'follow', 'learn', 'four', 'zero', 'go', 'down', 'six', 'visual', 'nine', 'one')
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model: Model = Model(
        kernels=(3, 3), strides=(2, 2), channels=(32, 32), hidden_size=16, activation='ReLU'
    )
    optim: Optim = Optim(
        lr=1e-3, n_epochs=10, batch_size=512
    )
    features: Features = Features(
        n_fft=400, win_length=400, hop_length=160, n_mels=64
    )
    augs: Augmentations = Augmentations(
        freq_mask_param=0, time_mask_param=0
    )
