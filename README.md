## Description

This code implements a quantum model for binary sentence classification. It is built on top of [lambeq](https://cqcl.github.io/lambeq/), a Python software library developed for the design and training of Quantum Natural Language Processing (QNLP) models. 

A [JAX](https://jax.readthedocs.io/en/latest/)-based backend is implemented to provide just-in-time (JIT) compilation and automatic differentiation capabilities. [qujax](https://github.com/CQCL/qujax), a JAX-compatible library for quantum circuit construction and evaluation, is used as a part of this backend. A class interfacing with the [Optax optimization library](https://optax.readthedocs.io/en/latest/) was developed, making most modern optimization algorithms available to use for training.


## Installation

To use the code included in this repository, you must first install a custom version of lambeq which has minor compatibility changes:

```pip install git+https://github.com/gamatos/lambeq.git@linear_model_changes```

Other dependencies include `qujax`, `optax`, `pandas`, `matplotlib`, `jax` and `jaxlib`, all of which can also be obtained via `pip`.

## Usage

You can see a full usage example in [`training_example.ipynb`](training_example.ipynb). 

A tokenised version of the [poem sentiment dataset](https://github.com/google-research-datasets/poem-sentiment) is provided in the [poem_sentiment](poem_sentiment) folder for example purposes only. For simplicity, all but the positive (here labeled with a 1) and negative (here labeled with a 0) examples have been removed. Note that this processed version of the dataset is distributed under the same [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) as the original, and this license differs from the rest of this repository.

## Technical note

In the language of [string diagrams](https://cqcl.github.io/lambeq/string_diagrams.html), used by lambeq, this model treats the sentences as if they had been first converted to diagrams containing only information about the order of the words using lambeq's [`LinearReader`](https://cqcl.github.io/lambeq/root-api.html?highlight=linearreader#lambeq.LinearReader) class.  Since the only relevant information is the word order, in practice this model works directly with tokenised sentences, and the diagram construction machinery is not needed. This, in conjunction with JAX, is used to dramatically speed up the evaluation of the model. 
