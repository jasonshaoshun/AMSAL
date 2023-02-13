# Erasure of Unaligned Attributes from Neural Representations

This repository contains the code for the experiments and algorithm from the paper [Erasure of Unaligned Attributes from Neural Representations](https://arxiv.org/abs/2302.02997) (appears at TACL 2023).

# Introduction

We present the Assignment-Maximization Spectral Attribute removaL (AMSAL) algorithm, which aims at removing information from neural representations when the information to be erased is implicit rather than directly being aligned to each input example.

It first loops between two Alignment and Maximization steps, during which it finds an alignment (A) based on existing projections and then projects the representations and guarded attributes into a joint space based on an existing alignment (M). After these two steps are iteratively repeated and an alignment is identified, the algorithm takes another step to erase information from the input representations based on the projections identified.

# Experimental Setting and Datasets

We use the Dataset and Experimental Settings from the paper

| Paper                                            | Github Link                                                  | Notes                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| [INLP](https://arxiv.org/abs/2004.07667)            | [View](https://github.com/shauli-ravfogel/nullspace_projection) | *Null it out: guarding protected attributes by iterative nullspsace projection*                              |
| [Bias-Bench](https://arxiv.org/abs/2110.08527)      | [View](https://github.com/mcgill-nlp/bias-bench)                | *An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models*           |
| [vulgartwitter](https://aclanthology.org/C18-1248/) | [View](https://github.com/ericholgate/vulgartwitter)            | *Expressively vulgar: The socio-dynamics of vulgarity and its effects on sentiment analysis in social media* |

# Datasets Download

The external datasets required by the projects above

Project| Dataset | Download Link | Notes | Download Directory
--------|--------|---------------|-------|-------------------
Bias-Bench | Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing)       | English Wikipedia dump used for SentenceDebias and INLP. (Section 4.3) | * `data/text`
INLP | Word Embedding      | [Download](download_data.sh) | GloVe Word Embeddings for Most Gendered Word (Section 4.1) | *
INLP | BiasBios            | [Download](download_data.sh) | BiasBios dataset for fair profession prediction (Section 4.1) | *

<!-- Bias-Bench| Wikipedia-10   | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing)       | English Wikipedia dump used for CDA and Dropout. | `data/text` -->

Each dataset should be downloaded to the specified path, relative to the root directory of the project.
