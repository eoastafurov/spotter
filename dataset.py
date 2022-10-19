from torch.utils.data import Dataset
import json
import torch 
import numpy 
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Type, Union, List
import pytorch_lightning as pl
from copy import deepcopy
from pathlib import Path
import pandas as pd
import torchaudio
import omegaconf

SEED = 43


class SpecScaler(torch.nn.Module):
    def forward(self, x):
        return torch.log(x.clamp_(1e-9, 1e9))


def collator(data):
    specs = []
    labels = []
    for wav, features, label in data:
        specs.append(features)
        labels.append(label)
    specs = torch.cat(specs)  
    labels = torch.Tensor(labels).long()
    return specs, labels


class SpotterDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            manifest_path: Path, 
            idx_to_keyword: List[str],
            transform, 
            ids: Optional[List[int]] = None
        ):
        super().__init__()
        self.transform = transform
        manifest = pd.read_csv(manifest_path)
        if ids is not None:
            manifest = manifest.loc[ids]
        self.wav_files = [
            manifest_path.parent / wav_path for wav_path in manifest.path
        ]
        keyword_to_idx = {
            keyword: idx for idx, keyword in enumerate(idx_to_keyword)
        }
        self.labels = [
            keyword_to_idx[keyword] for keyword in manifest.label
        ]
        
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.wav_files[idx])
        features = self.transform(wav)
        wav = wav[0]
        wav = wav / wav.abs().max()
        return wav, features, self.labels[idx]


class SpotterDataManager(pl.LightningDataModule):
    def __init__(
        self, 
        conf: omegaconf.dictconfig.DictConfig,
        num_workers,
        manifest_path: str = '/home/eugeny/Datasets/keyword-spotting/train/train/manifest.csv'
    ):
        super().__init__()
        self.conf = conf
        self.batch_size = conf.optim.batch_size
        self.num_workers = num_workers
        self.manifest_path = manifest_path

    def setup(self, stage: Optional[str] = None):
        # Init transforms
        train_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=self.conf.sample_rate, **self.conf.features),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=self.conf.augs.freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=self.conf.augs.time_mask_param),
            SpecScaler()
        )
        val_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=self.conf.sample_rate, **self.conf.features),
            SpecScaler()
        )

        # Init dataset
        # DatasetClass = SpotterDataset if not self.conf.experimantal.pretrain else GoogleSpeechCommands 
        dataset = SpotterDataset(
            manifest_path=Path(self.manifest_path),
            idx_to_keyword=self.conf.idx_to_keyword,
            transform=torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=self.conf.sample_rate, **self.conf.features),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=self.conf.augs.freq_mask_param),
                torchaudio.transforms.TimeMasking(time_mask_param=self.conf.augs.time_mask_param),
                SpecScaler()
            )
        )

        # Perform train/val split
        val_count = int(len(dataset) * self.conf.val_fraction)
        ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))
        val_ids = ids[:val_count]
        train_ids = ids[val_count:]

        # Init train/val datasets
        self.train_dataset = SpotterDataset(
            manifest_path=Path(self.manifest_path),
            idx_to_keyword=self.conf.idx_to_keyword,
            transform=train_transform,
            ids=train_ids
        )

        self.val_dataset = SpotterDataset(
            manifest_path=Path(self.manifest_path),
            idx_to_keyword=self.conf.idx_to_keyword,
            transform=val_transform,
            ids=val_ids
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator
        )
