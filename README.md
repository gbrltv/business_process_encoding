# Business Process Encoding

> Detection of anomalous business process traces in a scarcity of labels scenario. For that, it uses the word2vec encoding in combination with one-class classification algorithms

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/gbrltv/business_process_encoding/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/gbrltv/ProcessAnomalyDetector)](https://img.shields.io/github/issues/gbrltv/business_process_encoding)
[![GitHub forks](https://img.shields.io/github/forks/gbrltv/ProcessAnomalyDetector)](https://github.com/forks/gbrltv/business_process_encoding)
[![GitHub stars](https://img.shields.io/github/stars/gbrltv/ProcessAnomalyDetector)](https://img.shields.io/github/stars/gbrltv/business_process_encoding)
[![GitHub license](https://img.shields.io/github/license/gbrltv/ProcessAnomalyDetector)](https://img.shields.io/github/license/gbrltv/business_process_encoding)
[![Twitter](https://img.shields.io/twitter/url?style=social)](https://twitter.com/intent/tweet?text=Using+Business%20Process+Encoding:&url=https://github.com/gbrltv/business_process_encoding)

## Table of Contents

- [Installation](#installation)
- [Experimental Setup](#experimental-setup)
- [Data Analysis](#data-analysis)
- [Contributors](#contributors)

## Installation

### Clone

Clone this repo to your local machine using

```shell
git clone https://github.com/gbrltv/business_process_encoding.git
```

### Requirements

TODO: add requirements, such as python version and libraries used

## Experimental Setup - Recriating results

### Preparation

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


### Calculate encoding quality

TODO: add R code


### Classification experiments

To simulate the classification experiments, simply run:

```shell
python3 classification.py
```

This experiment uses a holdout of 80/20 for train/test. It reads the encodings from `encoding_results` and uses the Random Forest classifier due to its robustness. The results are saved on the `results.csv` file, which is uploaded in this repository.

## Data Analysis

TODO: link notebooks with plot generations and data analysis

## Contributors

- [Gabriel Marques Tavares](https://www.researchgate.net/profile/Gabriel_Tavares6), PhD candidate at Università degli Studi di Milano
- [Paolo Ceravolo](https://www.unimi.it/en/ugov/person/paolo-ceravolo), Associate Professor at Università degli Studi di Milano
- [Sylvio Barbon Junior](http://www.barbon.com.br/), Associate Professor at State University of Londrina
