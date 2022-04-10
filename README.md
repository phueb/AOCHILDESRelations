# AOCHILDESRelations

## About

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

### Corpus Data

The AO-CHILDES corpus used here was generated using an outdated version of the Python package [AOCHILDES](https://github.com/UIUCLearningLanguageLab/AOCHILDES) in 2018. 

The corpus is available in the folder `corpus` for users wishing to train and evaluate their own models.