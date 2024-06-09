#!/usr/bin/env python3
# coding: utf-8

"""
Adrien Gresse 2024

This script implement the training process and the evaluation of the model. 
usage: python3 ./traintest.py <data_folder> <results_folder>

Optionnal arguments control batch size, number of epoch (you can
see the full list of parameters with the --help flag).
"""

import time
import argparse
from os import path, stat
from functools import partial
from pprint import pprint
import copy
import json
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchaudio.backend.sox_io_backend import load

import numpy as np
import pandas as pd

from speechbrain.utils.metric_stats import BinaryMetricStats
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio

from utils import AudioConfig, AudioDataset, padded_batch, create_checkpoint
from models import AudioModel, ModelConfig


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("data_folder", help="where to look for json files")
    cmd_parser.add_argument("save_folder", help="where to store results")
    cmd_parser.add_argument("--batch-size", default=2, type=int)
    cmd_parser.add_argument("--max-epochs", default=25, type=int)
    cmd_parser.add_argument("--lr-init", default=1.0, type=float)
    cmd_parser.add_argument("--norm-mean", type=float, default=None)
    cmd_parser.add_argument("--norm-var", type=float, default=None)
    cmd_parser.add_argument("--target-size", type=int, default=2048)
    cmd_parser.add_argument("--use-energy", action="store_true", default=False)
    cmd_parser.add_argument("--num-mel-bins", default=40, type=int)
    cmd_parser.add_argument("--display-charts",
        action="store_true", default=False)
    args = vars(cmd_parser.parse_args())

    # -----------
    # PREPARATION
    # -----------

    # Creation of the AudioDataset objects (see utils.py) from JSON metadata
    # Training data are splited into train/validation sets.
    json_data_train = path.join(args["data_folder"], "vad_data_train.json") 
    json_data_eval = path.join(args["data_folder"], "vad_data_eval.json") 

    audio_conf = AudioConfig(
        use_energy=args["use_energy"],
        num_mel_bins=args["num_mel_bins"],
        target_size=args["target_size"]
    )

    audio_dataset = AudioDataset(json_data_train, audio_conf)
    audio_dataset_eval = AudioDataset(json_data_eval, audio_conf)

    train_dataset, val_dataset = random_split(audio_dataset, [0.75, 0.25])

    # Configuring the dataloader for data iteration
    # We use a custom collate function for Dataloader in order to perform
    # padding to the target size.
    padded_collate = partial(padded_batch, config=audio_conf)
    train_loader = DataLoader(train_dataset,
        batch_size=args["batch_size"], shuffle=True, collate_fn=padded_collate)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"],
        collate_fn=padded_collate, shuffle=True, drop_last=True)

    # Instantiating the model with optimizer and objective function
    model_conf = ModelConfig(
        input_size=args["num_mel_bins"] + args["use_energy"])
    audio_model = AudioModel(model_conf)

    init_model_path = path.join(args["save_folder"], "model_init.pth") 
    audio_model.save(init_model_path)

    device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    audio_model.to(device)

    # Relying on Adadelta optimizer here for simplicity sake
    optimizer = torch.optim.Adadelta(audio_model.parameters(), args["lr_init"])

    # Using binary cross-entropy since VAD is a binary classification problem
    criterion = nn.BCEWithLogitsLoss()

    logs = {
        "loss": [],
        "val_loss": []
    }

    # --------
    # TRAINING
    # --------

    for epoch in range(1, args["max_epochs"]+1):

        audio_model.train()

        losses_ = []
        # Optimizing model parameters on the training data
        for iter, batch in enumerate(train_loader):
            ids, x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Perform input normalization first
            if args["norm_mean"] and args["norm_var"]:
                x = (x - args["norm_mean"]) / args["norm_var"]

            # Forward pass
            predictions = audio_model(x)
            loss = criterion(predictions.squeeze(-1), y)
            losses_ += [loss.item()]

            # Computing gradient with backward pass and update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (iter + 1) % 10 == 0:
                print(
                    "Epoch {0:>3d}/{1}: iter {2:>4d}/{3} - loss={4:>5f}".format(
                        epoch,
                        args["max_epochs"],
                        iter+1,
                        len(train_loader),
                        loss.item()
                    ),
                    end="\r"
                )

        logs["loss"] += [np.mean(losses_)]

        # ----------
        # VALIDATION 
        # ----------

        # Performing validation step at end of the epoch
        # We use predictions to compute an average loss over
        # the validation data to monitor the overwhole training process
        print("\nValidation: ", end="")
        audio_model.eval()
        losses_ = []
        for iter, batch in enumerate(val_loader):
            ids, x, y = batch

            x = x.to(device)
            y = y.to(device)

            # Perform input normalization first
            if args["norm_mean"] and args["norm_var"]:
                x = (x - args["norm_mean"]) / args["norm_var"]

            predictions = audio_model(x)
            val_loss = criterion(predictions.squeeze(-1), y)
            losses_ += [val_loss.item()]

        logs["val_loss"] += [np.mean(losses_)]
        print(f"epoch {epoch} - val_loss={logs['val_loss'][-1]:.6f}", end="\n")

        # When achieving a better validation loss we save the model parameter.
        if logs["val_loss"][-1] < min(logs["val_loss"][:-1]+[np.inf]):
            audio_model.save(path.join(args["save_folder"], f"best_model.pth"))

        # Create a checkpoint of model parameters and optimizer
        create_checkpoint(
            path.join(args["save_folder"], f"checkpoint_{epoch}.pth"),
            audio_model,
            optimizer,
            epoch
        )

    # At end of training epochs we store our metrics
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(path.join(args["save_folder"], "logs.csv"))

    fig, axe = plt.subplots(1, 1)
    logs.index.name = "epochs"
    sns.lineplot(logs, ax=axe)
    plt.ylabel("Binary cross-entropy with logits loss")
    plt.tight_layout()
    plt.savefig(path.join(args["save_folder"], "training_history.pdf"))
    if args["display_charts"]:
        plt.show()

    # ----------
    # EVALUATION
    # ----------

    # Now performing evaluation on unseen data
    print("\n---------------------")
    print("Evaluation:")

    # Using custom padding function for technical reason. However we must note
    # that zero padded frames are not relevant for the evaluation and also
    # articially augments TN metrics.
    eval_loader = DataLoader(audio_dataset_eval, batch_size=args["batch_size"],
        collate_fn=padded_collate, shuffle=False, drop_last=True)

    # Restoring best performing model according to validation
    best_model_path = path.join(args["save_folder"], "best_model.pth")
    model_state = torch.load(
        best_model_path if path.exists(best_model_path) else init_model_path
    )
    audio_model.load_state_dict(model_state)
    audio_model.to(device)

    audio_model.eval()

    metrics = BinaryMetricStats(positive_label=1)
    losses_ = []
    predictions_ = []
    targets_ = []
    for iter, batch in enumerate(eval_loader):
        ids, x, y = batch

        x = x.to(device)
        y = y.to(device)

        # Performing input normalization
        if args["norm_mean"] and args["norm_var"]:
            x = (x - args["norm_mean"]) / args["norm_var"]

        predictions = audio_model(x)
        predictions = predictions.squeeze(-1)
        val_loss = criterion(predictions, y)

        losses_ += [val_loss.item()]
        predictions_ += [predictions.clone().detach().cpu()]
        targets_ += [y.clone().detach().cpu()]

        metrics.append(
            copy.deepcopy(ids),
            F.sigmoid(predictions.clone()).flatten(),
            y.clone().flatten()
        )

    targets_ = torch.concat(targets_)
    predictions_ = torch.concat(predictions_)

    # Saving prediction/targets for further analysis.
    np.savetxt(
        path.join(args["save_folder"], "predictions.csv"),
        predictions_.numpy()
    )
    np.savetxt(
        path.join(args["save_folder"], "targets.csv"),
        targets_.numpy()
    )

    # Computing binary classification metrics
    statistics = metrics.summarize()
    print("\n".join([k+": "+str(v) for k, v in statistics.items()]))
    metrics.clear()
    
    with open(path.join(args["save_folder"], "eval_stats.json"), "w") as fd:
        json.dump(statistics, fd, indent=1)

    # Trace the ROC curve
    RocCurveDisplay.from_predictions(
        targets_.numpy().flatten(),
        F.sigmoid(predictions_).numpy().flatten(),
        pos_label=1.0,
        name="Frame-level speech classification",
        c="k"
    )
    chance_lvl = np.linspace(0, 1, 100)
    plt.plot(chance_lvl, chance_lvl, "y--", label="Chance level")
    plt.legend()
    plt.savefig(path.join(args["save_folder"], "eval_roc_curve.pdf"))
    plt.tight_layout()
    if args["display_charts"]:
        plt.show()
    plt.show()

    # Display and example of results
    # Plot the predicted probabilities for a single random sample
    id_, x_, y_ = audio_dataset_eval[
        random.choice(range(len(audio_dataset_eval)))]
    wav, sr = load(
        path.join("../vad_data", id_ + ".wav"),
        normalize=True,
        channels_first=False
    )
    time_res = int(audio_conf.frame_shift * 1e-3 * audio_conf.sample_frequency)
    w_ = np.array([wav[i].item() for i in range(0, len(wav), time_res)])
    x_ = x_.unsqueeze(0).to(device)
    p_ = F.sigmoid(audio_model(x_)).detach().cpu().numpy().squeeze()

    fig, axe = plt.subplots(1, 1)
    axe.plot(w_, alpha=0.9, linewidth=2, c="tab:grey")
    axe.plot(y_, "tab:blue", alpha=0.7)
    axe.plot(p_, c="tab:orange", linewidth=2, alpha=0.8)
    axe.fill_between(range(len(y_)), y_, linestyle=":", color='tab:blue', alpha=0.2)

    plt.xlabel("Audio frames")
    plt.ylabel("Probability of speech")
    plt.legend(
        ["Audio waveform", "Groundtruth", "Model prediction"],
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig(path.join(args["save_folder"], "sample_prediction.pdf"))
    if args["display_charts"]:
        plt.show()
    plt.show()

