#!/usr/bin/env python3

"""
Adrien Gresse 2024

The script performs data preparation. It lists all files present in the data
folder and separate them into an evaluation and a training dataset in the form
of two distinct JSON files.

usage: python3 prepare_vad_data.py <data_folder> <output_folder>
Optionnal arguments control the split ratio and the isolation of speakers.
"""

import math
import copy
import itertools
import argparse
import os
from os import path
import json
from typing import Iterable, Dict, List, Tuple
import random


def list_all(folder: str) -> List[str]:
    """
    List data identifiers contained in the data folder ensuring every item
    comes with a .wav and .json file.
    Args:
        -folder: where to look for data
    Return:
        the list of all string data identifiers
    """
    all_files = {
        path.splitext(path.basename(item))[0]
        for item in os.listdir(folder)
    }
    for elem in all_files:
        assert path.isfile(path.join(folder, elem+".wav")) and \
                path.isfile(path.join(folder, elem+".json")), \
                f"missing file for id {elem}"
    return list(all_files)


def load_json(filename: str) -> Dict[str, List]:
    """
    Load the data dict coresponding to the given json file.
    Args:
        - filename: the path string to the json file
    Return:
        a dict with decoded data
    """
    with open(filename, "r") as fd:
        content = json.load(fd)
    return content


def save_json(json_dict: Dict, file_path: str):
    """
    Save the data dict into a JSON file.
    Args:
        -json_dict: the dict containing data
        -file_path: where to save data
    """
    with open(file_path, "w") as fd:
        json.dump(json_dict, fd, indent=1)


def create_speaker_index(ids: List[str]) -> Dict[str, List[str]]:
    """
    Given a data identifiers list, the function maps all speakers to their
    associated utterances (spk2utt).
    Args:
        -ids: the list of data identifiers
    Return:
        a dict mapping speaker id to their utterances
    """
    map_index = {}
    split_fields_fn = lambda x: path.splitext(x)[0].split("-")
    for i, (spk_id, sess_id, utt_id) in enumerate(map(split_fields_fn, ids)):
        #print(spk_id)
        if map_index.get(spk_id):
            map_index[spk_id] += [ids[i]]
        else:
            map_index[spk_id] = [ids[i]]
    return map_index


def sanitize_boundaries(boundaries: List[List[float]], tol: float=2e-3) -> List[List[float]]:
    """
    Clean segment shorter than threshold and avoid any overlaping segments
    Args:
        -boundaries: a list of boundaries [start, end]
        -tol: a float that define the minimal length of a segment in second
    """
    boundaries_ = copy.deepcopy(boundaries)
    short_segment_ids: List[int] = []

    for i, (start, end) in enumerate(boundaries):
        if end - start <= tol:
            short_segment_ids += [i]

        if i + 1 < len(boundaries):
            d = boundaries[i+1][0] - end
            if d < 0:
                boundaries_[i][1] -= 1e-3
                boundaries_[i+1][0] += 1e-3

    return [boundaries_[i] for i in range(len(boundaries_)) if i not in short_segment_ids]


def get_boundaries(json_speech: List[Dict]) -> List[List[float]]:
    """
    Extract and parse speech boundaries from a list of speech segments.
    Args:
        -json_speech: a list of speech segments boundaries as dict of strings
    Return:
        a list of boundaries parsed as floats
    """
    boundaries = []
    for segment in json_speech:
        boundaries += [
            [float(segment["start_time"]), float(segment["end_time"])]
        ]
    return boundaries


def create_json_data(ids: List[str], data_folder: str) -> Dict[str, Dict]:
    """
    Create a json dict from a list of data identifiers associating every data
    items to their audio path and speech segments boundaries.
    Args:
        -ids: the list of data identifiers
        -data_folder: where to look for the data files
    Return:
        a json dict with associated data
    """
    json_dict = {}
    for k in ids:
        file_path = path.join(data_folder, k+".json")
        annotations = load_json(file_path)
        speech = get_boundaries(annotations["speech_segments"])

        cleaned_speech = sanitize_boundaries(speech)
        
        json_dict[k] = {
            "wav": path.join(path.abspath(data_folder), k+".wav"),
            "speech": cleaned_speech
        }
    return json_dict


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("data_folder", help="Give the data location")
    cmd_parser.add_argument("output", help="Set the output folder")
    cmd_parser.add_argument("--eval-ratio", default=0.3, type=float, help="Ratio for evaluation")
    cmd_parser.add_argument("--isolate-speaker", action="store_true",
        default=False, help="Whether to isolate evaluation speakers or not")
    cmd_args = vars(cmd_parser.parse_args())

    ids = list_all(cmd_args["data_folder"])

    eval, train = [], []

    if cmd_args["isolate_speaker"]:
        print("Speaker isolation requested")

        spk_index = create_speaker_index(ids)
        spk_index_ = list(spk_index.keys())
        random.shuffle(spk_index_)

        eval_size = math.floor(cmd_args["eval_ratio"] * len(spk_index_))

        eval = list(itertools.chain.from_iterable([
            spk_index[k] for k in spk_index_[:eval_size]]))
        train = list(itertools.chain.from_iterable([
            spk_index[k] for k in spk_index_[eval_size:]]))

        assert set(eval) & set(train) == set(), "contamined split"

    else:
        eval_size = math.floor(cmd_args["eval_ratio"] * len(ids))
        random.shuffle(ids)

        eval = ids[:eval_size]
        train = ids[eval_size:]

    train_json = create_json_data(train, cmd_args["data_folder"])
    eval_json = create_json_data(eval, cmd_args["data_folder"])

    save_json(train_json, path.join(cmd_args["output"], "vad_data_train.json"))
    save_json(eval_json, path.join(cmd_args["output"], "vad_data_eval.json"))
