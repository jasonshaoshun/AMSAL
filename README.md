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
Bias-Bench | Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing)       | English Wikipedia dump used for SentenceDebias and INLP. (Section 4.3) | `data/text`
INLP | Word Embedding      | [Download](download_data.sh) | GloVe Word Embeddings for Most Gendered Word (Section 4.1) | `data/word-embedding`
INLP | BiasBios            | [Download](download_data.sh) | BiasBios dataset for fair profession prediction (Section 4.1) | `data/biography`

<!-- Bias-Bench| Wikipedia-10   | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing)       | English Wikipedia dump used for CDA and Dropout. | `data/text` -->

Each dataset should be downloaded to the specified path, relative to the root directory of the project.

# Datasets Size

This dataset and model size

Section | Experiment                                         | Model                                | Dataset size                  | Model Size
--------|----------------------------------------------------|--------------------------------------|-------------------------------|-------------------
4.1     | Word Embedding Debiasing                           | GloVe                                | 18M                           | 1 M
4.2     | BiasBios                                           | BertModel                            | 340M                          | 432 M
~       | ~                                                  | FastText                             | 134M                          | 164 M
4.3     | BiasBench                                          | BertModel + ALBERT + RoBERTa + GPT-2 | >> 1G (Provided by BiasBench) | 257M
4.4     | Multiple-Guarded Attribute Sentiment               | BertModel                            | 55M                           | 76 M
4.4     | Method Limitation (sentiment analysis on twitter)  | DeepMoji                             | 244M                          | 332 M



# Experiments

Suppose each dataset are downloaded or created to the specified path.

## Word Embedding Debiasing (Section 4.1 in the paper)

Move the script file under the folder of [script_glove](src/matlab/script_glove) to the root of [matlab](src/matlab/). Then call the script file to run AMSAL with specified hyperparameters, or call another script file to queue the experimental scripts automatically by the following code.

```sh
./src/matlab/matscript.sh
```

Please note: you need to change the start index and end index in [matscript.sh](src/matlab/matscript.sh), depends on how many matlab jobs you want to submit at once.

Check the spreadsheet created under the folder [data](data), named full.xlsx and epoch.xlsx.

## BiasBios Experiments (Section 4.2 in the paper)

### AM Step
Move the script files under the folder of [biography](src/matlab/script_biography/) to the root of [matlab](src/matlab/), then run AMSAL with hyperparameters specified in the scripts by calling the scripts one by one or automatically queueu the jobs by calling the following code. 

```sh
./src/matlab/matscript.sh
```

Check the result in the spreadsheet created under the folder of [data](data), named full.xlsx and epoch.xlsx.

Please also check the script files under the folder of [BertModel_PARTIAL](src/matlab/script_biography_BertModel_different_partial_n) and [FastTextModel_PARTIAL](src/matlab/script_biography_FastText_different_partial_n/) to run the partially supervised assignment of AMSAL.

### Removal Step and Downstream Tasks Evaluation

Run the script files to debias the neural representations of the biographies on genders, and perform the profession classifications.


```sh
./src/assignment/tpr-gap_biasbios.sh
```

### Export the results to spreadsheet and overleaf code

```sh
./src/assignment/export_tpr-gap_biography.sh
```
The tables will be stored in the folder of [tables](src/assignment/tables/) and the overleaf code will be stored in the folder of [biography](src/assignment/tables/biography/).




## BiasBench Experiments (Section 4.3 in the paper)


### Create Dataset

```sh
./src/bias-bench/batch_jobs/create_data.sh
```

### Alignment and Maximization step

Move the script files under the folder of [biasbench](src/matlab/script_biasbench/) to the root of [matlab](src/matlab/), then run AMSAL with hyperparameters specified in the scripts by calling the scripts one by one or automatically queueu the jobs by calling the following code. 

```sh
./src/matlab/matscript.sh
```

Check the result in the spreadsheet created under the folder of [data](data), named full.xlsx and epoch.xlsx.

### Removal Step

```sh
# Compute the projection matrix by SAL
./src/bias-bench/batch_jobs/sal_eigenvector.sh

# Compute the projection matrix by INLP
./src/bias-bench/batch_jobs/inlp_projection_matrix.sh
```

### Downstream Tasks

#### Task 1 Stereoset (section 4.3 in the paper)

```sh
# Evaluates non-debiased models against StereoSet
./src/bias-bench/batch_jobs/stereoset.sh

# Evaluates debiased models against StereoSet
./src/bias-bench/batch_jobs/stereoset_debias.sh

# Summary the results
./src/bias-bench/batch_jobs/stereoset_evaluation.sh

# Export to LaTeX code
./src/bias-bench/batch_jobs/export_stereoset.sh
```

Please check the spreadsheet and results json file under the folder of [stereoset](src/bias-bench/results/stereoset/), also the LaTex code stored in the folder of [tables](src/bias-bench/tables).


#### Task 2 CrowS-Pairs (section 4.3 in the paper)

```sh
# Evaluates non-debiased models against CrowS-Pairs
./src/bias-bench/batch_jobs/crows.sh

# Evaluates debiased models against CrowS-Pairs
./src/bias-bench/batch_jobs/crows_debias.sh

# Export to LaTeX code
./src/bias-bench/batch_jobs/export_crows.sh
```

Please check the results saved in the folder of [CrowS-Pairs](src/bias-bench/results/crows/) and LaTex code stored in the folder of [tables](src/bias-bench/tables).

#### Task 3 SEAT (Appendix B)


```sh
# Evaluates non-debiased models against SEAT
./src/bias-bench/batch_jobs/seat.sh

# Evaluates debiased models against SEAT
./src/bias-bench/batch_jobs/seat_debias.sh

# Export to LaTeX code
./src/bias-bench/batch_jobs/export_seat.sh
```

Please check the results saved in the folder of [SEAT](src/bias-bench/results/seat/) and LaTex code stored in the folder of [tables](src/bias-bench/tables).



#### Task 4 GLUE (Appendix B)

Very Complex, will be explained in a separate page

## Twitter Sentiment with Multiple Guarded Attributes Experiments (Section 4.4 in the paper)


### Create Dataset

```sh
# Save the Twitter Sentiment (Political) datasets in npz and matlab readable format
# Twitter dataset with original labelling
./src/assignment/create_twitter_dataset_.sh

# Twitter dataset labelled by word overlap
./src/assignment/create_twitter_dataset_labelled_by_overlap_.sh
```



## Deepmoji Experiments (Section 4.5 in the paper)

Deepmoji experiments consider the stereotypes of race in the twitters.

### AM Step
Move the script files under the folder of [deepmoji](src/matlab/script_deepmoji/) to the root of [matlab](src/matlab/), then run AMSAL with hyperparameters specified in the scripts by calling the scripts one by one or automatically queueu the jobs by calling the following code. 

```sh
./src/matlab/matscript.sh
```

Check the result in the spreadsheet created under the folder of [data](data), named full.xlsx and epoch.xlsx.

Please also check the script files under the folders of [R5S5](src/matlab/script_deepmoji_R5S5_different_partial_n/), [R5S8](src/matlab/script_deepmoji_R5S8_different_partial_n/) and [R8S5](src/matlab/script_deepmoji_R8S5_different_partial_n/) to run the partially supervised assignment of AMSAL.


### Removal Step and Downstream Tasks Evaluation

Run the script files to debias the neural representations of the biographies on genders, and perform the profession classifications.


```sh
./src/assignment/tpr-gap_deepmoji.sh
```

### Export the results to spreadsheet and overleaf code

```sh
./src/assignment/export_tpr-gap_table-deepmoji.sh
```

The tables will be stored in the folder of [tables](src/assignment/tables/) and the overleaf code will be stored in the folder of [deepmoji](src/assignment/tables/deepmoji/).
