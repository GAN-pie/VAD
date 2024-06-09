#!/usr/bin/env python3
# coding: utf-8


"""
Adrien Gresse 2024

This module contains the different pytorch modules that compose our model.
We can use the ModelConfig to modify the configuration of architecture.
"""

from typing import Union, Any, List, Dict, Tuple
from collections import namedtuple

import torch
from torch import nn, Tensor


ModelConfig = namedtuple(
    "ModelConfig",
    field_names=["input_size", "cnn_dim", "rnn_dim", "rnn_layers", "dnn_dim",
        "output_dim"],
    rename=False,
    defaults=[41, 32, 32, 2, 32, 1]
    # defaults=[41, 256, 128, 2, 128, 1]    # Larger model give similar results
)


class ConvBlock(nn.Module):
    """
    ConvBlock is a classical convolutional block. It expects a 3-D Tensor in
    its input [N, L, C] where N is the size of the batch, L the length of the
    sequence and C the dimension of the features. ConvBlock encodes the input
    features.
    """
    def __init__(
        self,
        input_size: int,
        channels: int,
        kernel_size: int=5,
        dropout: float=0.1,
        device: Union[torch.device, None] = None
    ):
        """
        Args:
            -input_size: define the dimension C of the input
            -channels: the output dimension
            -kernel_size: define the size of the kernel, expect a single value
                since block performs 1-D conv.
            -dropout: a float
            -device: the divice used for the computation
        """
        super().__init__()

        self.device = device

        self.conv1 = nn.Conv1d(
            input_size,
            channels,
            kernel_size,
            padding=2,
            padding_mode="reflect",
            device=device
        )
        self.norm1 = nn.LayerNorm(channels, device=device)
        self.activ1 = nn.LeakyReLU()

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=2,
            padding_mode="reflect",
            device=device
        )
        self.norm2 = nn.LayerNorm(channels, device=device)
        self.activ2 = nn.LeakyReLU()
        self.dropout = nn.Dropout1d(dropout)

        # We use Kaiming initialization to init filters
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x.to(self.device)
        x = x.transpose(1, -1)
        x = self.conv1(x)
        x = x.transpose(1, -1)
        x = self.norm1(x)
        x = self.activ1(x)
        x = x.transpose(1, -1)
        x = self.conv2(x)
        x = x.transpose(1, -1)
        x = self.norm2(x)
        x = self.activ2(x)
        x = self.dropout(x)
        x = x.squeeze(1)
        return x


class FullyConnectedBlock(nn.Module):
    """
    FullyConnectedBlock used for final classification
    """
    def __init__(
        self,
        input_size: int,
        n_units: int,
        dropout: float=0.1,
        device: Union[torch.device, None] = None
    ):
        """
        FullyConnectedBlock is expecting [*, C] input shape.
        Args:
            -input_size: an int defining the size of the input
            -n_units: an int defining the dimension of the output
            -dropout: specifies the dropout rate
            -device: where the computation accurs
        """
        super().__init__()

        self.device = device

        self.linear = nn.Linear(input_size, n_units, device=device)
        self.activ = nn.LeakyReLU()
        self.norm = nn.BatchNorm1d(n_units, device=device)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x.to(self.device)
        x = self.linear(x)
        x = self.activ(x)
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(1, -1)
        x = self.dropout(x)
        return x


class AudioModel(nn.Module):
    """
    AudioModel is composed of 3 different blocks. A ConvBlock that encodes the
    features, a RNN that learns from the temporality of the sequence and a 
    final FullyConnectedBlock for classification. It expects 3-D input Tensors
    of shape [N, L, C].
    """
    def __init__(
        self,
        config: Union[ModelConfig, None]=None,
        device: Union[torch.device, None] = None
    ):
        """
        Args:
            -config: a ModelConfig instance, or None for default config
            -device: a torch.device where graph computation will occur
        """
        super().__init__()

        self.device = device
        self.cfg = config if config is not None else ModelConfig()

        self.prenorm = nn.LayerNorm(self.cfg.input_size, device=device)

        self.cnn = nn.Sequential(
            ConvBlock(self.cfg.input_size, self.cfg.cnn_dim//2),
            ConvBlock(self.cfg.cnn_dim//2, self.cfg.cnn_dim)
        )

        self.rnn = nn.GRU(
            self.cfg.cnn_dim,
            self.cfg.rnn_dim,
            self.cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            device=device
        )

        self.dnn = nn.Sequential(
            FullyConnectedBlock(self.cfg.rnn_dim*2, self.cfg.dnn_dim),
            FullyConnectedBlock(self.cfg.dnn_dim, self.cfg.dnn_dim),
            nn.Linear(self.cfg.dnn_dim, 1, bias=False, device=device)
        )

        # We use orthogonal initialization which is better for hidden state
        # parameters of RNN. Other weights are initialized with Kaiming.
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l1)
        nn.init.kaiming_normal_(self.rnn.weight_ih_l0)
        nn.init.kaiming_normal_(self.rnn.weight_ih_l1)

    def forward(self, x: Tensor) -> Tensor:
        x.to(self.device)
        x = self.prenorm(x)
        x = self.cnn(x)
        x, _ = self.rnn(x)
        x = self.dnn(x)
        return x

    def save(self, filepath: str, preserve_views: bool=True):
        """
        Save the model parameters
        Args:
            -filepath: a str for the storage location
            -preserve_views: whether to save relations between module's tensors
                or not, if False it can decrease the volume on disk
        """
        if preserve_views:
            torch.save(self.state_dict(), filepath)
        else:
            torch.save(
                {
                    n: p.clone() for n, p in self.named_parameters()
                },
                filepath
            )


if __name__ == "__main__":

    model_config = ModelConfig()
    audio_model = AudioModel(model_config)
    
    trainable_params = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in audio_model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    x = torch.rand(1, 2048, 41)
    x = audio_model(x)

