#!/usr/bin/env python3
# coding: utf-8

"""
Adrien Gresse 2024

The module provides the VoiceActivityDetector class

Usage: python3 vad.py <model_folder> <waveformed_file_path>
"""


import itertools
import os
from os import path
import argparse
from typing import Union, List, Dict, Tuple
# from itertools import batched

import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance.kaldi import fbank
from torchaudio.backend.sox_io_backend import load

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.polynomial as P
import scipy.interpolate as I
from scipy import ndimage

from models import AudioModel, ModelConfig
from utils import AudioConfig


class VoiceActivityDetector:
    """
    The VoiceActivityDetector class provides a user friendly interface to
    perform speech activity detection.
    """
    def __init__(
        self,
        model_src: str,
        save_folder: str,
        th_speech: float=0.55,
        th_noise: float=0.3,
        energy_vad: bool=False
    ):
        """
        Args:
            -model_src: define the path of the VAD model to be used
            -save_folder: where the speech boundaries will be stored
            -th_speech: define the speech activity decision threshold
            -th_noise: define the noise decision threshold
        """
        assert path.isfile(model_src), "invalid model"
        os.makedirs(save_folder, exist_ok=True)

        if energy_vad:
            self.audio_cfg = AudioConfig(num_mel_bins=0, use_energy=True)
        else:
            self.audio_cfg = AudioConfig()

        self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")

        if energy_vad:
            self.model_cfg = ModelConfig(
                input_size=self.audio_cfg.num_mel_bins \
                        + self.audio_cfg.use_energy
            )
        else:
            self.model_cfg = ModelConfig()
        self.model = AudioModel(self.model_cfg)
        self.model.load_state_dict(
            torch.load(model_src, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.th_speech = th_speech
        self.th_noise = th_noise

    def speech_probs(self, wav: Tensor, sr: int) -> Tensor:
        """
        Frame-level speech probabilities predicted with the VAD model.
        Args:
            -wav: the waveform with Pytorch like shape (N, )
            -sr: the sampling frequency of the audio
        Return:
            the frame-level speech raw probabilities
        """
        assert sr == self.audio_cfg.sample_frequency, "invalid sampling rate"

        feats = fbank(
            wav,
            energy_floor=self.audio_cfg.energy_floor,
            frame_length=self.audio_cfg.frame_length,
            frame_shift=self.audio_cfg.frame_shift,
            num_mel_bins=self.audio_cfg.num_mel_bins \
                    if self.audio_cfg.num_mel_bins else 40,
            raw_energy=self.audio_cfg.raw_energy,
            sample_frequency=self.audio_cfg.sample_frequency,
            use_energy=self.audio_cfg.use_energy
        )

        # TODO: better normalization
        feats = (feats - feats.mean()) / feats.std()

        if self.audio_cfg.num_mel_bins == 0:
            feats = feats[:, 0][:, None]

        feats = feats.to(self.device)
        predictions = self.model(feats.unsqueeze(0)).squeeze()
        predictions = F.sigmoid(predictions)

        return predictions.detach().cpu()

    def apply_smoothing(
        self,
        predictions: Tensor,
        mode: str="kalman",
        smoothing_factor:float=0.92
    ) -> Tensor:
        """
        Apply smoothing over the raw frame-level raw probabilities. For better
        results the speech activation/deactivation thresholds might be tuned
        accordingly.
        Args:
            -predictions: frame-level speech probabilities
            -mode: set the smoothing method, possible choices are exponential
                moving average (\"emm\"), kalman filtering (\"kalman\")
            -smoothing_factor: only relevant for emm smoothing
        Return:
            a Tensor with smoothed probabilities frame-level decisions
        """
        preds = pd.Series(predictions.clone().numpy())
        
        if mode == "emm":
            assert smoothing_factor >= 0 and smoothing_factor < 1.0, \
                    "invalid smoothing factor"

            preds = preds.ewm(alpha=1-smoothing_factor, adjust=False).mean()
            return Tensor(preds.values)

        # TODO: not satisfying, to be improved (use wavelet correction method?)
        # if mode == "spline":
        #     t = np.linspace(0, len(preds), len(preds))
        #     s = I.make_interp_spline(t, preds, k=3)
        #     #preds = [splev(x, s) for x in range(len(preds))]
        #     preds = I.splev(preds, (s.t, s.c, 3))
        #     print(preds)
        # elif mode == "poly":
        #     t = np.linspace(0, len(preds), len(preds))
        #     p = np.poly1d(np.polyfit(t, preds, 15))
        #     preds = [p(x) for x in range(len(preds))]

        elif mode == "kalman":
            # Simple implementation of Kalman filter from wikipedia
            n_iter = len(preds)
            size = (n_iter,)
            R = 0.01**2
            Q = 1e-5*(1-smoothing_factor)

            x_hat = np.zeros(size)
            g = np.zeros(size)
            x_hat_ = np.zeros(size)
            g_ = np.zeros(size)
            K = np.zeros(size)
            x_hat[0] = 0.0
            g[0] = 1.0

            for k in range(1, n_iter):
                x_hat_[k] = x_hat[k-1]
                g_[k] = g[k-1] + Q

                K[k] = g_[k] / (g_[k] + R)
                x_hat[k] = x_hat_[k] + K[k] * (preds[k] - x_hat_[k])
                g[k] = (1 - K[k]) * g_[k]
            preds = x_hat
        return Tensor(preds)

    def compute_decision(self, predictions, smooth:bool=False):
        """
        Compute the decision from proababilities, where 0 = non-speech
        a where 1 = speech.
        Args:
            -predictions: frame-level probabilities
            -smooth: whether to perform decision smoothing or not
        Return:
            a Tensor with the frame-level decisions
        """
        above_th = predictions >= self.th_speech
        below_th = predictions > self.th_noise
        decisions = above_th.to(int) + below_th.to(int)
        decisions = pd.Series(decisions.clone().numpy())
        decisions = decisions.rolling(24).min()
        # print(decisions.value_counts())
        decisions[decisions  < 1] = 0
        decisions[decisions >= 1] = 1

        return self._smooth_decisions(Tensor(decisions)) \
                if smooth else Tensor(decisions)

    def _smooth_decisions(self, decisions: Tensor, n_iter:int=1) -> Tensor:
        """
        Perform decision smoothing based on a morphological dilation.
        Args:
            -decisions: frame-level speech decisions
            -n_iter: number of iterations of the morphological operation
        Return:
            a smoothed frame-level speech decisions Tensor
        """
        closed_decisions = ndimage.binary_dilation(
            decisions.clone().numpy(), iterations=n_iter).astype("int32")
        return Tensor(closed_decisions)

    def _merge_filter_segments(
        self,
        boundaries: Tensor,
        dur_th: float=0.25,
        merge_th: float=0.022
    ) -> Tensor:
        """
        Merging segments when they are too close and remove isolated segments
        when they are too short.
        Args:
            -bondaries: speech segments boundaries
            -dur_th: minial duration for a speech segment should be in
                the range [0.15-0.25]
            -merge_th: minimal duration of a nonspeech segment should lies in
                the range [0.01-0.02]
        """
        boundaries_ = []
        merged = []
        for i in range(2, len(boundaries), 2):
            diff = boundaries[i] - boundaries[i-1]
            if diff <= merge_th:
                merged += [i, i-1]

        boundaries_ = [
            boundaries[i].item()
            for i in range(0, len(boundaries)) if i not in merged
        ]

        final_boundaries = []
        filtered = []
        for i in range(1, len(boundaries_), 2):
            diff = boundaries_[i] - boundaries_[i-1]

            if diff <= dur_th:
                filtered += [i-1, i]

        final_boundaries = [
            boundaries_[i]
            for i in range(0, len(boundaries_)) if i not in filtered
        ]

        return Tensor(final_boundaries)

    def derive_boundaries(self, decisions: Tensor) -> Tensor:
        """
        Derive the speech semgent boundaries from the frame-level decisions.
        Args:
            -decisions: frame-level speech decisions
        Return:
            speech segment boundaries in sequencial order
        """
        decisions[0] = 0
        decisions[-1] = 0
        time_res = self.audio_cfg.frame_shift * 1e-3 \
                * self.audio_cfg.sample_frequency

        speech_frames = (decisions == 1).int().nonzero()

        boundaries = []
        for i in speech_frames:
            if decisions[i] != decisions[i-1] \
                    or decisions[i] != decisions[i+1]:
                boundaries += [i]
        boundaries = torch.cat(boundaries)

        speech = self._merge_filter_segments(
            boundaries*time_res/self.audio_cfg.sample_frequency)

        return speech


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("model_source", help="path to a pytorch Module")
    cmd_parser.add_argument("audio_file", help="path to target audio file")
    cmd_parser.add_argument("--save_folder",
        help="where to store results", default="./results_vad")
    cmd_parser.add_argument("--activation-threshold", type=float, default=0.55)
    cmd_parser.add_argument(
        "--deactivation-threshold", type=float, default=0.3)
    #cmd_parser.add_argument("--energy-vad", action="store_true", default=False)
    cmd_parser.add_argument("--smoothing",
        choices=["emm", "kalman"],
        default="kalman")
    cmd_parser.add_argument("--energy-vad", help="use only energy",
        action="store_true", default=False)

    args = vars(cmd_parser.parse_args())

    vad = VoiceActivityDetector(
        args["model_source"],
        args["save_folder"],
        args["activation_threshold"],
        args["deactivation_threshold"],
        energy_vad=args["energy_vad"]
    )

    try:
        wav, sr = load(
            args["audio_file"],
            normalize=True,
            channels_first=True
        )
    # The sox backend might cause error depending on torchaudio version.
    except OSError:
        wav, sr = torchaudio.load(
            args["audio_file"],
            normalize=True,
            channels_first=True
        )

    time_res = vad.audio_cfg.frame_shift * 1e-3 \
            * vad.audio_cfg.sample_frequency

    probs = vad.speech_probs(wav, sr)

    smooth_probs = vad.apply_smoothing(probs, mode=args["smoothing"])

    #plt.plot(probs, "b")
    #plt.plot(smooth_probs, "r")
    #plt.show()

    decisions = vad.compute_decision(smooth_probs)

    boundaries = vad.derive_boundaries(decisions)

    #plt.plot(decisions)
    #plt.vlines(boundaries*vad.audio_cfg.sample_frequency/time_res, 0, 0.5, colors="r")
    #plt.show()

    results_path = path.join(
        args["save_folder"],
        path.splitext(path.basename(args["audio_file"]))[0] + ".txt"
    )
    with open(results_path, "w") as fd:
        for i in range(0, len(boundaries) - 1, 2):
            fd.write(f"{boundaries[i]}  {boundaries[i+1]} speech\n")

    # Plot the VAD results
    wav = wav.squeeze(0)
    w_ = np.array([wav[i].item() for i in range(0, len(wav), int(time_res))])

    b_ = boundaries * vad.audio_cfg.sample_frequency / time_res

    fig, axe = plt.subplots(1, 1)
    l1, = axe.plot(w_, alpha=0.8, linewidth=2, c="tab:blue")
    l2, = axe.plot(probs, c="tab:orange", linewidth=2, alpha=0.8)
    l3, = axe.plot(decisions, c="tab:grey", linestyle=":", linewidth=2, alpha=0.7)
    lines = axe.vlines(b_, 0., 1., colors="tab:red", linewidth=2)

    plt.legend(
        [l1, l2, l3, lines],
        ["Audio waveform (downsampled)", "Model probabilities", "Frame-level speech decision",
            "Speech segments boundaries"],
        loc="lower right"
    )

    axe.fill_between(
        range(len(decisions)), decisions, color='tab:grey', alpha=0.2)

    plt.xlabel("Audio frames")
    plt.ylabel("Probability of speech")
    plt.title(path.splitext(path.basename(args["audio_file"]))[0])
    plt.tight_layout()
    plt.show()
