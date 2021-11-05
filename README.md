# AOCHILDESRelations

## Background

This repository contains words extracted from the AO-CHILDES corpus,
and information about the semantic relations they enter with each other.

## Relations included in the data

- Hypernymy
- Cohyponymy
- Meronymy
- Attribute
- Synonymy
- Antonymy

The data is located in the folder `relations`. Each line in each text file contains several words. 
The first word is the target concept, and the remaining words are the "relata" (i.e words that are related to the target concept).


## Two-Process Theory

Given word embeddings for words in AO-CHILDES (learned by any distributional semantic model),
it is possible to "fine-tune" the word embedding space using a supervised learning algorithm based on the data provided in this repository. 

This is the procedure used to test the two-process theory of semantic development,
outlined in a CogSci 2019 submission, available [here](https://osf.io/6jfkx/).

The results and statistical analyses used in the paper are available in the folder `TwoProcessTheory`
The original code repository referenced in the paper is no longer available due to incompatibility with newer version of `Ludwig`,
required for submitting jobs to the UIUC Language & Learning job submission system.

The task used in Process-2, after training Process-1 models, to fine-tune word embeddings is called "identification".
It is similar to a multiple-choice task, where the Process-2 model is tasked with identifying one correct relatum out of several lures.

Different process-2 architectures were explored. They vary slightly, but essentially perform the same task:

- Comparator
- Classifier
- Extractor 
- Aligner

The code that implements these models is available in `TwoProcessTheory/architectures`

### Corpora 

There are two different CHILDES corpora in the repository used as input to the word embedding models. 
`childes-20171212.txt` was generated in the same way as `childes-20180319.txt` except that a few additional steps were taken:
1) all titlecased strings were replaced with a single symbol ("TITLED")
2) all words tagged by the Python package spacy as referring to a person or organization were replaced by a single symbol ("NAME_B"if the word is the first in a span of words referring to a person or organization, and "NAME_I" if it is not the first word in a span of words referring to a person or organization)
