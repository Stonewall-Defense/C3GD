# Certus Caliber Classification Gunshot Dataset (C3GD)

> [Overview](#certus-caliber-classification-gunshot-dataset-c3gd) | [Download](#download) | [Results](#results) | [Repository Content](#repository-content) | [License](#license) | [Citing](#citing)
>
> [![LICENSE](https://img.shields.io/badge/license-CC%20BY-blue.svg)](#license)&nbsp;[![DOWNLOAD](https://img.shields.io/badge/download-.zip-ff69b4.svg)](#download)

We introduce the Certus Caliber Classification Gunshot Dataset (C3GD), a publicly accessible data set developed for the analysis of firearm muzzle blast sounds. It aims to provide a wide variety of firearms, calibers, cartridges, microphones, and microphone locations with metadata detailed beyond what is currently otherwise available. The dataset comprises more than 8000 field-collected data points from 28 firearms across 16 calibers. Data collection in the field is costly, so much research has been done using gunshot audio collected from the internet, which increases the risk of low-quality data and label noise. This dataset is primarily focused on caliber classification, but can also be used for gunshot detection, audio separation, and audio signal processing. The dataset aims to provide enough diversity in the dataset to be able to generalize to more real-world applications while providing enough metadata for detailed academic analysis.

## Download

The dataset can be downloaded as a single .zip file (~430 MB): **[Download the C3GD dataset](FIXME)**

## Results

The suitability of the data for deep learning was verified by preliminary training runs. Using the [TIMM](https://timm.fast.ai/) implementation of the ubiquitous ResNet architecture, we trained a model to over 97% test accuracy in a few hours. Our parameters were as follows:

- Features: single channel, log-scaled mel spectrogram (see [Raponi, Oligeri, and Ali (2021)](https://arxiv.org/pdf/2004.07948))
- Architecture: ResNet-18
- Training: LR 0.0005, 100 epochs

## Repository Content

- [`data/*.wav`](data/)

  8015 audio clips of live outdoor gunshots

  File naming convention: `{ClassId}-{EventId}-{Platform}-{Mic}-{FileId}-{ClipId}.wav`

  - `{ClassId}` - Unique identifier of the class, referenced to [`classes.json`](classes.json)
  - `{EventId}` - Denotes the collection event, referenced to [`metadata/events.csv`](metadata/events.csv)
  - `{Platform}` - Denotes the platform (gun) used for the shot, referenced to [`metadata/platforms.csv`](metadata/platforms.csv)
  - `{Mic}` - Denotes the mic used to collect the audio. Mic stats are found in [`metadata/microphones.csv`](metadata/microphones.csv), and their location at a given `EventId` is recorded in [`metadata/microphone_locations.csv`](metadata/microphone_locations.csv).
  - `{FileId}` - Denotes the opaque UID of the original, unclipped audio file.
  - `{ClipId}` - Denotes the clip number from a specific `FileId` and `Mic` combination. Because clips denote subsequent shots with the same platform, they are expected to cause pseudoreplication only when both the `ClipId` and `FileId` match for different `Mic` values (that is, they are different recordings of the same gunshot), not when only the `FileId` matches.

- [`metadata/*.csv`](metadata/)

  Detailed metadata not usually found in gunshot audio datasets

  - [`calibers.csv`](metadata/calibers.csv) - Details for each caliber (class) used
    - Rimfire ammunition is generally not classified as pistol, rifle, etc.
  - [`cartridges.csv`](metadata/cartridges.csv) - Details for each cartridge used
    - Note that "cartridges" are commonly referred to as "bullets"
    - Shotgun ammunition is generally given in oz, while other calibers are given in grains
    - Shotgun ammunition loses speed quickly depending on the loading, so it has no clear supersonic/subsonic designation
    - One cartridge, denoted as `unk_556NATO_77`, was not properly labeled during the test
  - [`metadata/events.csv`](metadata/events.csv) - Details for each data collection event
  - [`metadata/microphones.csv`](metadata/microphones.csv) - Specifications for each microphone used to collect data
    - Note that while some microphones can collect above 48 kHz, all data in this repository has been resamples to 48 kHz
  - [`metadata/microphone_locations.csv`](metadata/microphone_locations.csv) - Distance and azimuth values for mic locations at each data collection event
  - [`metadata/platforms.csv`](metadata/platforms.csv) - Details for each platform (gun) used
    - Note that for custom or modified platforms, not all details are available

- [`scripts/*.py`](scripts/)

  Miscellaneous Python scripts for post-processing metadata. Run these from this directory.

- [`classes.json`](classes.json)

  A list of classes (calibers) for testing classification results

- [`metadata.csv`](metadata.csv)

  Per-file metadata records, sufficient to train a classifer. New and non-obvious columns are described below.

  - `channel_orientation` - For mics that record in stereo, denotes the left or right channel
  - `is_phone` - Initial results suggest that phone-recorded data is notably different from high-quality mics; you may want to treat them as separate populations
  - `day` and `part` - The `oh_farm` event was conducted in morning and afternoon sessions over two days with significantly different weather
  - `is_silenced` - Denotes the use of a silencer

- [`seraph.json`](seraph.json)

  Control metdata for the [Seraph](https://github.com/Stonewall-Defense/libseraph) multimedia dataset management tool

## License

The dataset is available under the terms of the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Citing

TODO
