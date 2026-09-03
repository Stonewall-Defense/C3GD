# Certus Caliber Classification Gunshot Dataset (C3GD)

> [Overview](#certus-caliber-classification-gunshot-dataset-c3gd) | [Download](#download) | [Results](#results) | [Repository Content](#repository-content) | [License](#license) | [Citing](#citing)
>
> [![LICENSE](https://img.shields.io/badge/license-CC%20BY-blue.svg)](#license)&nbsp;[![DOWNLOAD](https://img.shields.io/badge/download-.zip-ff69b4.svg)](#download)

In this work, we introduce the Certus Caliber Classification Gunshot Dataset (C3GD), a publicly accessible dataset developed for the analysis of firearm muzzle blast sounds. The dataset aims to provide a wide variety of firearms, calibers, cartridges, microphones, and microphone locations with metadata detailed beyond what is currently available. It comprises more than 12,000 live-fire data points from 45 firearms across 18 calibers, including nine calibers not present in existing datasets. Live-fire data collection in semi-controlled environments is costly, but using internet-collected data increases the risk of low-quality data and label noise. This dataset focuses primarily on caliber classification, but can also facilitate gunshot detection, audio separation, and audio signal processing applications, providing a diverse and realistic reference. It aims to provide enough diversity to generalize to real-world applications while also providing enough metadata for detailed academic analysis.

## Download

The dataset can be downloaded from Zenodo (~771 MB): **[Download the C3GD dataset](https://zenodo.org/records/22286299)**

## Results

The suitability of the data for deep learning was verified by preliminary training runs. Using the [TIMM](https://timm.fast.ai/) implementation of the ubiquitous ResNet architecture, we trained a model to over 97% test accuracy in a few hours. Our parameters were as follows:

- Features: single channel, log-scaled mel spectrogram (see [Li et al. (2022)](https://www.mdpi.com/2079-9292/11/23/3859))
- Architecture: ResNet-18
- Training: LR 0.0005, 100 epochs

See [below](#repository-content) for training code.

## Repository Content

- [`data/*.wav`](data/)

  12112 audio clips of live outdoor gunshots

  File naming convention: `{ClassId}-{EventId}-{Platform}-{MicId}-{FileId}-{ClipId}.wav`

  - `{ClassId}` - Unique identifier of the class, referenced to [`classes.json`](classes.json)
  - `{EventId}` - Denotes the collection event, referenced to [`metadata/events.csv`](metadata/events.csv)
  - `{Platform}` - Denotes the platform (gun) used for the shot, referenced to [`metadata/platforms.csv`](metadata/platforms.csv)
  - `{MicId}` - Denotes the mic used to collect the audio. Mic stats are found in [`metadata/microphones.csv`](metadata/microphones.csv), and their location at a given `EventId` is recorded in [`metadata/microphone_locations.csv`](metadata/microphone_locations.csv).
  - `{FileId}` - Denotes the opaque UID of the original, unclipped audio file.
  - `{ClipId}` - Denotes the clip number from a specific `FileId` and `Mic` combination. Because clips denote subsequent shots with the same platform, they are expected to cause pseudoreplication only when both the `ClipId` and `FileId` match for different `Mic` values (that is, they are different recordings of the same gunshot), not when only the `FileId` matches.

- [`metadata/*.csv`](metadata/)

  Detailed metadata not usually found in gunshot audio datasets

  - [`calibers.csv`](metadata/calibers.csv) - Details for each caliber (class) used
    - **Note:** Rimfire ammunition is generally not classified as pistol, rifle, etc.
  - [`cartridges.csv`](metadata/cartridges.csv) - Details for each cartridge used
    - Note that "cartridges" are commonly referred to as "bullets"
    - Shotgun ammunition is generally given in oz, while other calibers are given in grains
    - Shotgun ammunition loses speed quickly depending on the loading, so it has no clear supersonic/subsonic designation
    - One cartridge, denoted as `unk_556NATO_77`, was not properly labeled during the test
  - [`metadata/events.csv`](metadata/events.csv) - Details for each data collection event
  - [`metadata/microphone_locations.csv`](metadata/microphone_locations.csv) - Distance and azimuth values for mic locations at each data collection event
  - [`metadata/microphones.csv`](metadata/microphones.csv) - Specifications for each microphone used to collect data
    - Note that while some microphones can collect above 48 kHz, all data in this repository has been resamples to 48 kHz
  - [`metadata/platforms.csv`](metadata/platforms.csv) - Details for each platform (gun) used
    - Note that for custom or modified platforms, not all details are available
  - [`metadata/weather.csv`](metadata/weather.csv) - Details for weather for each day of each event
    - Because events can last for several hours, the mean value for each collection time period is reported based on [historical data](https://www.wunderground.com/)

- [`scripts/*.py`](scripts/)

  Python scripts for showing how the dataest was created, visualized, and benchmarked. Run these from this (the root) directory.

  - [`example_clip.py`](scripts/example_clip.py) - Clip gunshots from a raw example file and show the outputs
  - [`example_features.py`](scripts/example_features.py) - Generate spectra for example data and visualize the outputs
  - [`full_training.py`](scripts/full_training.py) - Train a benchmark model on the dataset for comparison/validation
    - This script requires one positional argument, `param_filename`
    - Two exmple parameter files are included as `scripts/params/*.json`
    - Example usage: `python3 scripts/full_training.py scripts/params/exp_augmented.json`

- [`classes.json`](classes.json)

  A list of classes (calibers) for testing classification results

- [`metadata.csv`](metadata.csv)

  Per-file metadata records, sufficient to train a classifer. New and non-obvious columns are described below.

  - `channel_orientation` - For mics that record in stereo, denotes the left or right channel
  - `is_phone` - Initial results suggest that phone-recorded data is notably different from high-quality mics; you may want to treat them as separate populations
  - `day` - The `oh_farm` and `nj_farm_2` events were conducted over two days with significantly different weather

- [`seraph.json`](seraph.json)

  Control metdata for the [Seraph](https://github.com/Stonewall-Defense/libseraph) multimedia dataset management tool

## License

The dataset is available under the terms of the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Citing

Gurny, Sinclair, and Ryan Quinn. “Descriptor: Certus Caliber Classification Gunshot Dataset (C3GD).” arXiv:2606.18135. Preprint, arXiv, June 16, 2026. https://doi.org/10.48550/arXiv.2606.18135.
