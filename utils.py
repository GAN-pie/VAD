#!/usr/bin/env python3
# coding: utf-8

"""
Adrien Gresse 2024

This module contains data utilites for training pipeline (Dataset, collate_fn).
In addition, the module can be used in standalone in order to compute some
statistics over the entire dataset (mean, std).

usage python3 ./utils.py
"""

from functools import partial
import json
from typing import NamedTuple, List, Dict, Tuple, Union
from collections import namedtuple
from pprint import pprint

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchaudio.backend.sox_io_backend import load
from torch.utils.data import Dataset
from torchaudio.transforms import MelScale, MFCC
from torchaudio.compliance.kaldi import fbank


AudioConfig = namedtuple(
    "AudioConfig",
    field_names=["frame_length", "frame_shift", "num_mel_bins", "target_size",
        "sample_frequency", "use_energy", "raw_energy", "energy_floor"],
    rename=False,
    defaults=[25.0, 10.0, 40, 2048, 16000, True, True, 0.1]
)


class AudioDataset(Dataset):
    """
    A Pytorch Dataset overloaded for audio data
    """
    def __init__(self, json_data: str, config: Union[AudioConfig, None]=None):
        """
        Args:
            -json_data: path to the JSON file describing dataset
            -config: an instance of AudioConfig or None for default config
        """
        self.cfg = config if config is not None else AudioConfig()

        with open(json_data, "r") as fd:
            self.data_dict = json.load(fd)

        self.index_dict = {i: k for i, k in enumerate(self.data_dict.keys())}

        self.num_speech_segments = sum([
            len(item["speech"]) for item in self.data_dict.values()])

        self.time_res = self.cfg.frame_shift * 1e-3 * self.cfg.sample_frequency

    def __len__(self) -> int:
        """
        Return:
            the length of the dataset (number of audio sequence)
        """
        return len(self.index_dict)

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor]:
        """
        Fetch a single sample from the dataset.
        Return:
            a tuple with sample identifier, features tensor and target labels
        """
        ids = self.index_dict[idx]
        datum = self.data_dict[ids]
        wav, sr = load(datum["wav"], normalize=True, channels_first=True)

        feats = fbank(
            wav,
            energy_floor=self.cfg.energy_floor,
            frame_length=self.cfg.frame_length,
            frame_shift=self.cfg.frame_shift,
            num_mel_bins=self.cfg.num_mel_bins \
                    if self.cfg.num_mel_bins else 40,
            raw_energy=self.cfg.raw_energy,
            sample_frequency=self.cfg.sample_frequency,
            use_energy=self.cfg.use_energy
        )

        win_len = self.cfg.frame_length * 1e-3 * self.cfg.sample_frequency
        hop_len = self.time_res

        target_size = ((len(wav[0, :]) - win_len) // hop_len) + 1
        assert target_size == feats.size(0), "mismatching lengths"

        ground_thrue = torch.zeros(int(target_size))
        for start, end in datum["speech"]:
            start_i = start * self.cfg.sample_frequency
            end_i = end * self.cfg.sample_frequency

            first_frame = start_i // self.time_res
            last_frame = end_i // self.time_res
            if end_i + win_len >= len(wav[0,:]):
                last_frame = target_size.size(-1)

            ground_thrue[int(first_frame):int(last_frame)] = 1.0

        if self.cfg.num_mel_bins == 0:
            feats = feats[:, 0][:, None]
        # print(feats.size())

        return ids, feats, ground_thrue


def padded_batch(
    tensors: List[Tuple[str, Tensor, Tensor]],
    config: AudioConfig
) -> Tuple[List[str], Tensor, Tensor]:
    """
    Pad all element in the batch to fixed target size (AudioConfig).
    Args:
        tensors: a batch of samples
        config: an instance of AudioConfig
    Return:
        a tuple of batched identifiers, features, and labels
    """

    channel_size = tensors[0][1].size(1)
    max_shape = (config.target_size, channel_size)  # perform a fixed padding
    # uncomment following lines to pad relatively to the batch
    # max_shape = (0, channel_size)
    # for i, x, y in tensors:
    #     assert channel_size == x.size(1), \
    #         f"mismatching channel size expect {channel_size}, got {x.size(1)}"
    #     if x.size(0) > max_shape[0]:
    #         max_shape = (x.size(0), channel_size)

    ids = []
    X = []
    Y = []
    for i, x, y in tensors:
        ids += [i]
        if x.size(0) < max_shape[0]:
            X += [F.pad(x, (0, 0, 0, max_shape[0] - x.size(0)))]
            Y += [F.pad(y, (0, max_shape[0] - y.size(0)))]
        else:
            X += [x[:max_shape[0], :]]
            Y += [y[:max_shape[0]]]

    return (ids, torch.stack(X, 0), torch.stack(Y, 0))


def create_checkpoint(
    filepath: str,
    model: nn.Module,
    opt: Union[torch.optim.Optimizer, None] = None,
    epoch: int=-1
):
    """
    Checkpointing model and optimiizer state allow the resuming of training in
    case of crash.
    Args:
        -filepath: path to checkpoint file
        -model: a torch.Module
        -opt: a torch.optim.Optimizer
        -epoch: the current epoch
    """
    if opt is None:
        torch.save(model.state_dict(), filepath)
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "last_epoch": epoch
            },
            filepath
        )


if __name__ == "__main__":

    # Computing the training data statistics and sequences length
    # These statistics are used for the normalization of the data with
    # --norm-mean and norm-var parameters.
    # In addition we compute the lengths of the sequence to find a good
    # trade-off for the target-size parameters.
    data_file = "metadata/vad_data_train.json"
    audio_conf = AudioConfig()
    audio_data = AudioDataset(data_file, audio_conf)

    means = []
    stds = []
    lengths = []
    for i in range(len(audio_data)):
        ids, feats, targets = audio_data[i]

        means += [feats.mean()]
        stds += [feats.std()]
        lengths += [feats.size(0)]
        print(ids, means[-1], stds[-1], lengths[-1])

    print(np.mean(means), np.mean(stds), np.quantile(lengths, [0.0, 0.25, 0.5, 0.75, 1.0]))
