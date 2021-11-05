# AOCHILDESRelations

## Background

This repository contains words extracted from the AO-CHILDES corpus,
and information about the relations they enter with each other.

## Relations included in the data

- Hypernymy
- Cohyponymy
- Meronymy
- Attribute
- Synonymy
- Antonymy


## Advanced

Given word embeddings for words in AO-CHILDES (learned by any distributional semantic model),
it is possible to "fine-tune" the word embedding space using a supervised learning algorithm based on the data provided in this repository. 

This is the procedure used to test the two-process theory of semantic development,
outlined in a CogSci 2019 submission, available [here](https://osf.io/6jfkx/).

The results and statistical analyses used in the paper are available in the folder `TwoProcessTheory`
The original code repository referenced in the paper is no longer available due to incompatibility with newer version of `Ludwig`,
required for submitting jobs to the UIUC Language & Learning job submission system.

### Matching
consists of matching a probe with multiple correct answers

### Identification
consists of identifying correct answer from multiple-choice question

## Architectures for learning how words are related

- Comparator
- Classifier

## Corpora 

There are two different CHILDES corpora in the repository used as input to the word embedding models. 
`childes-20171212.txt` was generated in the same way as `childes-20180319.txt` except that a few additional steps were taken:
1) all titlecased strings were replaced with a single symbol ("TITLED")
2) all words tagged by the Python package spacy as referring to a person or organization were replaced by a single symbol ("NAME_B"if the word is the first in a span of words referring to a person or organization, and "NAME_I" if it is not the first word in a span of words referring to a person or organization)
