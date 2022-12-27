## Description
This code implements a word-sequence model (i.e. one where the only syntactic information is the ordering of the words) for quantum binary sentence classification. 

It is meant to be used alongside [lambeq](https://cqcl.github.io/lambeq/), a Python software library developed for the design and training of Quantum Natural Language Processing (QNLP) models. 

In the language of [string diagrams](https://cqcl.github.io/lambeq/string_diagrams.html), used by lambeq, this model treats the sentences as if they had been first converted to diagrams containing only information about the order of the words using lambeq's [`LinearReader`](https://cqcl.github.io/lambeq/root-api.html?highlight=linearreader#lambeq.LinearReader) class.  Since the only relevant information is the word order, in practice this model works directly with tokenised sentences, and the diagram construction machinery is not needed.

A [JAX](https://jax.readthedocs.io/en/latest/)-based backend is used to provide just-in-time (JIT) compilation and automatic differentiation capabilities. These are used in conjunction with the simple structure of the diagrams to dramaticaly speed up the evaluation of the model. [qujax](https://github.com/CQCL/qujax), a JAX-compatible library for quantum circuit construction and evaluation, is used as a part of this backend. 

Also included is a class interfacing with the [Optax optimization library](https://optax.readthedocs.io/en/latest/), making most modern optimization algorithms available to use for training.

## Installation

To use the code included in this repository, you must first install a custom version of lambeq which has minor compatibility changes:

```pip install git+https://github.com/gamatos/lambeq.git@linear_model_changes```

Other dependancies include `qujax`, `optax`, `pandas`, `matplotlib`, `jax` and `jaxlib`, all of which can also be obtained through `pip`.

## Usage

You can see a full usage example in [`training_example.ipynb`](training_example.ipynb). 

A tokenised version of the [poem sentiment dataset](https://github.com/google-research-datasets/poem-sentiment) is provided in the [poem_sentiment](poem_sentiment) folder as a toy example for training. For simplicity, all but the positive (here labeled with a 1) and negative (here labeled with a 0) examples have been removed. Note that this processed version of the dataset is distributed under the same [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) as the original, and this license differs from the rest of this repository.