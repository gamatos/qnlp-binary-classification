"""
LinearModel
==========
A specialized lambeq model that evaluates sentences for which the
corresponding diagram is assumed to have a linear structure (i.e. as
if they had been created through the `LinearReader` class).

This is leveraged so that JAX can efficiently batch the evaluation
and so that we can reuse the function that evaluates the circuits.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import jax
import numpy


def build_linear_circuit(start : jax.numpy.ndarray,
                         word_circuit : Callable[[jax.numpy.ndarray,
                                                  jax.numpy.ndarray],
                                                 jax.numpy.ndarray],
                         combining_circuit : Callable[[jax.numpy.ndarray,
                                                       jax.numpy.ndarray,
                                                       jax.numpy.ndarray],
                                                      jax.numpy.ndarray],
                         word_weight_nr : int,
                         end : Callable[[jax.numpy.ndarray],
                                        jax.numpy.ndarray] = lambda _: _):
    """
    Builds a circuit that evaluates a linear diagram from the
    JAX-compatible circuits corresponding to the word type and
    the combining diagram.

    Parameters
    ----------
    start : jax.numpy.ndarray
        A statevector representing the initial state of the circuit
    word_circuit : Callable
        The circuit corresponding to the word type
    combining_circuit : Callable
        The circuit corresponding to the combining diagram
    word_weight_nr : Callable
        The number of parameters the word circuit takes
    end : Callable
        Function to be applied to the output of the circuit
    """

    # Applies one instance of the combining diagram
    def apply_step(left, angles):
        initial_left = left
        pad_flag = angles[-1]
        right = word_circuit(left, angles[:word_weight_nr])
        left = combining_circuit(left, right,
                                 angles[word_weight_nr:-1])

        left = jax.lax.select(jax.numpy.full(left.shape,
                                             pad_flag,
                                             dtype=bool),
                              left, initial_left)
        return left, None

    # Evaluates full circuit
    def evaluate_circuit(angles: jax.numpy.ndarray):
        # Apply first word circuit
        left_init = word_circuit(start,
                                 angles[0, :word_weight_nr])

        # Ignore it if it corresponds to padding
        left_init = jax.lax.select(jax.numpy.full(left_init.shape,
                                                  angles[0, -1],
                                                  dtype=bool),
                                   left_init, start)

        # The jax.lax.scan call is equivalent to:
        # x = left_init
        # for a in angles[1:, :]:
        #     x, _ = apply_step(x, a)
        # res = x

        # Successively combine all diagrams
        res, _ = jax.lax.scan(apply_step, left_init, angles[1:, :])

        res = end(res)

        return res

    return evaluate_circuit


class LinearModel():
    """A specialized lambeq model for training of linear diagrams."""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.predictions(*args, **kwds)

    def __init__(self,
                 word_list: Iterable[str],
                 max_sentence_length: int,
                 start: jax.numpy.ndarray,
                 word_circuit: Callable,
                 combining_circuit: Callable[[jax.numpy.ndarray,
                                              jax.numpy.ndarray,
                                              jax.numpy.ndarray],
                                             jax.numpy.ndarray],
                 word_weight_nr: int,
                 combining_weight_nr: int,
                 end: Callable[[jax.numpy.ndarray,
                                jax.numpy.ndarray],
                               jax.numpy.ndarray] = lambda _: _,
                 normalise: bool = False,
                 **kwargs) -> None:
        """Initialise a LinearModel.

        Parameters
        ----------
        word_list : Iterable[str]
            List of words that make up the model
        max_sentence_length : int
            The maximum admissible sentence length
            Used for padding purposes
        start : jax.numpy.ndarray
            A statevector representing the initial state of the circuit
        word_circuit : Callable
            The circuit corresponding to the word type
        combining_circuit : Callable
            The circuit corresponding to the combining diagram
        word_weight_nr : Callable
            The number of parameters the word circuit takes
        combining_weight_nr : Callable
            The number of parameters the combining circuit
        end : Callable
            The initial state of the circuit
        normalise : bool
            Whether to normalise output state

        """
        word_set = set(word_list)
        # Add unknown token to represent words not seen in training set
        word_set.add("UNK")
        # Mapping from words to indices
        self.word_dict = {w: i for i, w in enumerate(word_set)}

        self.max_sentence_length = max_sentence_length
        self.word_circuit = word_circuit
        self.combining_circuit = combining_circuit
        self.word_weight_nr = word_weight_nr
        self.combining_weight_nr = combining_weight_nr
        self.normalise = normalise

        # Builds circuit corresponding to diagram structure
        circuit = build_linear_circuit(start,
                                       word_circuit,
                                       combining_circuit,
                                       word_weight_nr,
                                       end)

        # Wrapper that includes normalisation in circuit evaluation
        def _evaluator(x, normalise):
            res = jax.vmap(circuit)(x)
            if normalise:
                res = LinearModel._normalise_vector(res)
            return res
        evaluator = jax.jit(_evaluator, static_argnames='normalise')

        self.circuit_evaluator = evaluator
        self.weights = jax.numpy.array([])

    def initialise_weights(self,
                           generator: Callable = numpy.random.rand) -> None:
        """
        Initialise the weights of the model.

            Parameters
        ----------
        generator : Callable, default = numpy.random.rand
            Callable that generates model's weigths

        """
        nr_of_words = len(self.word_dict)

        self.weights = jax.numpy.array(generator(nr_of_words
                                                 * self.word_weight_nr
                                                 + self.combining_weight_nr))

    def _batched_weight_indices(self,
                                tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns the parameters
        needed to evaluate the circuit as a single array to be supplied
        to the batched evaluation function.
        """

        # Get list of indices corresponding to the words in each
        # sentence. If word index can not be found, index for token
        # representing unknown word is used instead
        indices = [[self.word_dict.get(w, self.word_dict["UNK"]) + 1 for w in ts]
                   for ts in tokenised_sentences]

        # Add indices corresponding to padding  
        for i in indices:
            i += [0] * (self.max_sentence_length - len(i))

        indices = numpy.array(indices)
        return indices

    def _indices_from_diagrams(self,
                               tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns indices in the
        array of model weights indicating which are relevant to
        evaluate those sentences
        """

        indices = sorted({self.word_dict[w]
                         for ts in tokenised_sentences for w in ts})

        indices = numpy.array(indices, dtype=int)

        return indices

    def _relevant_word_parameter_mask(self,
                                      tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns a bit mask to the
        array of word weights indicating which are relevant to
        evaluate those sentences
        """

        # Create and apply mask
        i = self._indices_from_diagrams(tokenised_sentences)
        mask = numpy.ones(self.word_weights.shape, dtype=int)
        mask[i, :] = 0

        return mask

    def _relevant_parameter_mask(self,
                                 tokenised_sentences: Iterable[Iterable[str]]):
        """
        Given a set of tokenised sentences, returns a bit mask to the
        stacked array of weights indicating which are relevant to
        evaluate those sentences
        """
        mask = self._relevant_word_parameter_mask(tokenised_sentences)
        mask = mask.reshape(-1)

        stairs_mask = numpy.zeros(self.combining_weights.shape, dtype=int)
        mask = numpy.hstack([mask, stairs_mask])

        return mask

    def _stacked_weights(self, weights : Optional[jax.numpy.ndarray] = None):
        """
        Joins weights corresponding to the word circuit and the
        combining circuit in a single array
        """

        if weights is None:
            weights = self.weights

        word_weights = self.get_word_weights(weights)
        combining_weights = self.get_combining_weights(weights)
        combining_weights = jax.numpy.repeat(
                        combining_weights.reshape(1, -1),
                        word_weights.shape[0], axis=0)

        weights = jax.numpy.hstack([word_weights, combining_weights])

        return weights

    def _padded_weights(self, stacked_weights: jax.numpy.ndarray = None):
        """
        Takes a set of weights represented by an array and returns a
        version that accounts for padding.

        This is done by adding a row corresponding to the padding token,
        and a column indicating which rows correspond to words and which
        is the padding token.
        """
        if stacked_weights is None:
            stacked_weights = self._stacked_weights()

        nr_of_words = len(self.word_dict)
        weights_per_layer = self.word_weight_nr + self.combining_weight_nr

        # Extra column/rows to deal with padding
        padded_weights = jax.numpy.c_[stacked_weights,
                                      jax.numpy.ones(nr_of_words)]
        padded_weights = jax.numpy.vstack([jax.numpy.zeros(weights_per_layer
                                                           + 1),
                                           padded_weights])

        return padded_weights

    def _indices_to_angles(self,
                           indices: jax.numpy.ndarray,
                           weights: Optional[jax.numpy.ndarray] = None):
        """
        Converts a set of word indices into the corresponding weights
        """
        if weights is None:
            padded_weights = self._padded_weights()
        else:
            stacked_weights = self._stacked_weights(weights)
            padded_weights = self._padded_weights(stacked_weights)

        return padded_weights[indices]

    def _predictions(self,
                     indices: jax.numpy.ndarray,
                     normalise: Optional[bool] = None,
                     weights: Optional[jax.numpy.ndarray] = None
                     ) -> jax.numpy.ndarray:
        """
        Perform forward pass of model from the indices in the
        weight array.
        """
        if normalise is None:
            normalise = self.normalise
        if weights is None:
            weights = self.weights

        x = self._indices_to_angles(indices, weights)
        res = self.circuit_evaluator(x, normalise)
        return res

    def predictions(self,
                    tokenised_sentences: Iterable[Iterable[str]],
                    normalise: Optional[bool] = None) -> jax.numpy.ndarray:
        """
        Generate model's predictions from a set of tokenised sentences

        Parameters
        ----------
        tokenised_sentences : Iterable[Iterable[str]]
            The sentences to be evaluated.
        normalise : Optional[bool]
            Whether the normalise the output

        Returns
        -------
        jax.numpy.ndarray
            Array containing model's prediction.
        """
        if normalise is None:
            normalise = self.normalise

        indices = self._batched_weight_indices(tokenised_sentences)
        x = self._indices_to_angles(indices)

        res = self.circuit_evaluator(x, normalise)
        return res

    @classmethod
    def from_tokenised_sentences(
                cls,
                tokenised_sentences: Iterable[Iterable[str]],
                start: jax.numpy.ndarray,
                word_circuit: Callable,
                combining_circuit: Callable,
                word_weight_nr: int,
                stair_weight_nr: int,
                end: Callable = lambda _: _,
                normalise: bool = False) -> jax.numpy.ndarray:
        """
        Create model from a set of tokenised sentences

        Parameters
        ----------
        start : jax.numpy.ndarray
            A statevector representing the initial state of the circuit
        word_circuit : Callable
            The circuit corresponding to the word type
        combining_circuit : Callable
            The circuit corresponding to the combining diagram
        word_weight_nr : Callable
            The number of parameters the word circuit takes
        combining_weight_nr : Callable
            The number of parameters the combining circuit
        end : Callable
            The initial state of the circuit
        normalise : bool
            Whether to normalise output state
        """

        words = [w for s in tokenised_sentences for w in s]
        return cls(words, max(map(len, tokenised_sentences)),
                   start, word_circuit, combining_circuit,
                   word_weight_nr, stair_weight_nr, end, normalise)

    @staticmethod
    def _normalise_vector(predictions: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Normalise diagram output."""
        predictions = jax.numpy.atleast_2d(predictions)
        predictions = jax.numpy.square(jax.numpy.abs(predictions))
        return predictions / predictions.sum(axis=1).reshape(-1, 1)

    def get_word_weights(self, weights: Optional[jax.numpy.ndarray] = None):
        """Get weights needed to evaluate to word circuits."""
        if weights is None:
            weights = self.weights
        i = len(self.word_dict) * self.word_weight_nr
        return weights[:i].reshape(len(self.word_dict), -1)

    def get_combining_weights(self,
                              weights: Optional[jax.numpy.ndarray] = None):
        """Get weights needed to evaluate combining circuit."""
        if weights is None:
            weights = self.weights
        i = self.combining_weight_nr
        if i == 0:
            return jax.numpy.array([])
        return self.weights[-i:]

    @property
    def word_weights(self) -> jax.numpy.ndarray:
        """Weights needed to evaluate word circuits."""
        return self.get_word_weights()

    @word_weights.setter
    def word_weights(self, new) -> None:
        ww = self.get_word_weights()
        assert new.shape == ww.shape
        ww[:] = new

    @property
    def combining_weights(self) -> jax.numpy.ndarray:
        """Weights needed to evaluate combining circuit."""
        return self.get_combining_weights()

    @combining_weights.setter
    def combining_weights(self, new) -> None:
        sw = self.get_combining_weights()
        assert new.shape == sw.shape
        sw[:] = new

    @property
    def symbols(self) -> Iterable[str]:
        """Model symbols (represented as strings)."""
        return list(self.word_dict.keys())

    def from_checkpoint():
        """Load model from checkpoint (not yet implemented!)."""
        raise NotImplementedError()

    # NOTE: The following methods are used to implement the
    # evaluation of a loss function from the model predictions. This
    # is currently a bit awkward because the resulting JIT'd functions
    # are evaluated in the context of the model, but it doesn't
    # necessarily make sense for the model object to own it.
    #  So these functions are returned and the helper method
    # `LinearModel.evaluate` is used to evaluate them on some data with
    # the model's weights. A better approach might be to create a class
    # from which we can instantiate a loss function object that is
    # linked to the model.

    def evaluate(
                self,
                f: Callable,
                tokenised_sentences: Iterable[Iterable[str]],
                *args,
                **kwargs) -> jax.numpy.ndarray:
        """
        Given a function f, evaluates it on the weights of the model
        and the word indices corresponding to the supplied tokenised
        sentences
        """
        indices = self._batched_weight_indices(tokenised_sentences)
        res = f(self.weights, indices, *args, **kwargs)
        return res

    def loss(self,
             loss: Callable[[jax.numpy.ndarray,
                             jax.numpy.ndarray],
                            float],
             normalise: Optional[bool] = None) -> Callable[[jax.numpy.ndarray,
                                                            jax.numpy.ndarray,
                                                            jax.numpy.ndarray],
                                                           float]:
        """
        Given a a function that evaluates the loss from the predictions
        of the model, returns one that evaluates it from its weights
        and the indices of the words making up the sentences
        """
        if normalise is None:
            normalise = self.normalise

        def loss_from_weights(w, i, y):
            y0 = self._predictions(i, normalise, w)
            res = loss(y0, y)
            return res

        return loss_from_weights

    def grad_loss(self,
                  loss: Callable = lambda _: _,
                  normalise: Optional[bool] = None):
        """
        Given a a function that evaluates the loss from the predictions
        of the model, returns one that evaluates the gradient
        """
        loss_from_weights = self.loss(loss, normalise)
        return jax.jit(jax.grad(loss_from_weights))
