# Business Process Encoding

> Evaluation of encodings in the context of business processes. In total, ten different encodings are evaluated. The encodings are divided into three categories: process mining based, text based and graph based.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/gbrltv/business_process_encoding/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/gbrltv/business_process_encoding)](https://img.shields.io/github/issues/gbrltv/business_process_encoding)
[![GitHub forks](https://img.shields.io/github/forks/gbrltv/business_process_encoding)](https://github.com/forks/gbrltv/business_process_encoding)
[![GitHub stars](https://img.shields.io/github/stars/gbrltv/business_process_encoding)](https://img.shields.io/github/stars/gbrltv/business_process_encoding)
[![GitHub license](https://img.shields.io/github/license/gbrltv/business_process_encoding)](https://img.shields.io/github/license/gbrltv/business_process_encoding)

## Table of Contents

- [Installation](#installation)
- [Experimental Setup](#experimental-setup)
  - [Data preparation](#data-preparation)
  - [Generate encodings](#generate-encodings)
  - [Measuring encoding quality](#measuring-encoding-quality)
  - [Classification experiments](#classification-experiments)
- [References](#references)
- [Contributors](#contributors)

## Installation

### Clone

Clone this repo to your local machine using

```shell
git clone https://github.com/gbrltv/business_process_encoding.git
```

## Experimental Setup

### Data preparation

Before running the experiments, it is necessary to convert the original logs (`csv`) to the `xes` format. For that, run:

```shell
python3 utils/convert_csv_to_xes.py
```

This code convert files under the `event_logs` folder and write them at `event_logs_xes`. This process is necessary to run a few encodings in the next step.


### Generate encodings

To generate the encodings, simply run the files under the `compute_encoding` folder. Example:

```shell
python3 compute_encoding/alignment.py
```

The results are saved under the `encoding_results` folder. Run all encodings needed for the analysis.


### Measuring encoding quality

To extract encoding quality metrics, run:

```shell
Rscript feature_metrics.R
```

The script reads the encodings from `encoding_results` and perform metrics extraction based on [1](https://aps.arxiv.org/abs/1808.10406v1) and [2](http://www.jmlr.org/papers/volume21/19-348/19-348.pdf). The results are saved in the `dataset_ALL.csv` file, also included in the repository.


### Classification experiments

To simulate the classification experiments, simply run:

```shell
python3 classification.py
```

This experiment uses a holdout of 80/20 for train/test. It reads the encodings from `encoding_results` and uses the Random Forest classifier due to its robustness. The results are saved on the `results.csv` file, which is uploaded in this repository.


## References

[Barbon Jr., S., Ceravolo, P., Damiani, E., Tavares, G.M.: Evaluating trace encoding methods in process mining, 2021](https://link.springer.com/chapter/10.1007/978-3-030-70650-0_11)


## Contributors

- [Gabriel Marques Tavares](https://www.researchgate.net/profile/Gabriel_Tavares6), PhD candidate at Università degli Studi di Milano
- [Paolo Ceravolo](https://www.unimi.it/en/ugov/person/paolo-ceravolo), Associate Professor at Università degli Studi di Milano
- [Sylvio Barbon Junior](http://www.barbon.com.br/), Associate Professor at State University of Londrina
