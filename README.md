# Business Process Encoding

> Extraction of encodings in from even logs. In total, 22 different encodings are implemented. The encodings are divided into four categories: baseline, process mining based, text based and graph based.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/gbrltv/business_process_encoding/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/gbrltv/business_process_encoding)](https://img.shields.io/github/issues/gbrltv/business_process_encoding)
[![GitHub forks](https://img.shields.io/github/forks/gbrltv/business_process_encoding)](https://github.com/forks/gbrltv/business_process_encoding)
[![GitHub stars](https://img.shields.io/github/stars/gbrltv/business_process_encoding)](https://img.shields.io/github/stars/gbrltv/business_process_encoding)
[![GitHub license](https://img.shields.io/github/license/gbrltv/business_process_encoding)](https://img.shields.io/github/license/gbrltv/business_process_encoding)

## Table of Contents

- [Installation](#installation)
- [Extract encodings](#extract-encodings)
- [Parameters](#parameters)
- [References](#references)
- [Contributors](#contributors)

## Installation

Clone this repo to your local machine using

```shell
git clone https://github.com/gbrltv/business_process_encoding.git
```

Create an environment and activate it

```shell
conda create --name bpe python=3.10
conda activate bpe
```

Install dependencies

```shell
python -m pip install -r requirements.txt
```

## Extract encodings

To generate the encodings, simply call the main function and provide the arguments. Example:

```shell
python main.py --dataset=event_logs/scenario1_1000_attribute_0.05.xes --encoding=onehot
```

The computed encodings are returned and then can be used for further tasks.

## Parameters

| Parameter | Description | Options |
| ------------- | ------------- | ------------- |
| dataset  | event log path  | - |
| encoding | encoding method used to extract embeddings | onehot, count2vec, alignment, logskeleton, tokenreplay, doc2vec, hash2vec, tfidf, word2vec, boostne, deepwalk, diff2vec, glee, grarep, hope, laplacianeigenmaps, netmf, nmfadmm, node2vec, nodesketch, role2vec, walklets |
| vector_size | number of desired dimensions for the encoding (note that some encoding methods do not allow to configure this option) | - |
| aggregation | how to aggregate activities' encodings to represent a complete trace (exclusive for some methods of the text and graph families of encodings) | average, max |
| embed_from | extract encodings from nodes or edges (exclusive for the graph-based encodings) | nodes, edges |
| edge_operator | how to aggregate edge embeddings (exclusive for the graph-based encodings) | average, hadamard, weightedl1, weightedl2 |

## References

[Barbon Jr., S., Ceravolo, P., Damiani, E., Tavares, G.M.: Evaluating trace encoding methods in process mining, 2021](https://link.springer.com/chapter/10.1007/978-3-030-70650-0_11)

[Tavares, G.M., Oyamada, R.S., Barbon Jr., S., Ceravolo, P.: Trace encoding in process mining: A survey and benchmarking, 2023](https://www.sciencedirect.com/science/article/pii/S0952197623012125)

## Contributors

- [Gabriel Marques Tavares](https://www.dbs.ifi.lmu.de/cms/personen/mitarbeiter/tavares/index.html), Postdoc at LMU München
- [Rafael Seidi Oyamada](https://sesar.di.unimi.it/staff/rafael-oyamada/) PhD candidate at Università degli Studi di Milano
- [Paolo Ceravolo](https://www.unimi.it/en/ugov/person/paolo-ceravolo), Associate Professor at Università degli Studi di Milano
- [Sylvio Barbon Junior](http://www.barbon.com.br/), Associate Professor at Univeristà degli Studi di Trieste
