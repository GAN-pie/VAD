# Voice Activity Detection

A from scratch implementation of voice activity detection using Pytorch.


You will find in this folder the code to reproduce all the results presented in the `vad_technical_report.pdf` file.


## VAD inferences
```
python vad.py <model_path> <wave_file>
```

This vad.py script allows user to control activation and deactivation threshold by passing a value between 0.0 and 1.0 to the `--activation-threshold` and `--deactivation-threshold` arguments.

Speech probabilities by default are smoothed thanks to a [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) but user can choose an Exponential Moving Average with the `--smoothing` argument.


## Model training
The complete training/evaluation procedure is organized into the run.sh file. All of the user need is to replace the ROOT variable with its own path and `chmod +x run.sh` then `./run.sh`.

If user want to train it manually, data (a subset of Librispeech) much first be downloaded. Then, user will have to create the metadata JSON files with the `prepare_vad_data.py` script.

Train and evaluation are performed by the `traintest.py` script.
```
python traintest.py <data_folder> <save_folder>
```

User can control different parameters, use the `--help` flag to see the complete list.

> For normalization, mean and standard deviation can be compute by calling the `utils.py` module as a standolone script.
