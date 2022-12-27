"""
OptaxOptimizer
=============
Module interfacing with the Optax optimisation library
(https://optax.readthedocs.io/en/latest/).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional
import warnings

import jax
import numpy as np
import optax

from lambeq.training.optimizer import Optimizer


class OptaxOptimizer(Optimizer):
    """
    Class interfacing with the Optax optimisation library.

    Used to provide JAX-based models with access to standard
    machine learning optimisation algorithms.
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn: Callable,
                 hyperparams: Optional[dict[Any, Any]] = None) -> None:
        """
        Initialise the Optax optimizer.

        Parameters
        ----------
        model : :py:class:`.Model`
            A JAX-based lambeq model.
        optimizer : optax.base.GradientTransformation
            An optax optimizer
        loss_fn : Callable
            A loss function of form `loss(prediction, labels)`.
        hyperparams : dict of str to float.
            A dictionary containing the optimizer's hyperparameters.
        """
        if hyperparams is None:
            hyperparams = dict()

        super().__init__(model, hyperparams, loss_fn)

        self.optimizer = optimizer(**hyperparams)
        optimizer_state = self.optimizer.init(model.weights)
        self.model = model
        self.optimizer_state = optimizer_state
        self.current_sweep = 1
        self.loss_fn = jax.jit(loss_fn)
        self.grad_loss = model.grad_loss(self.loss_fn)

    def backward(
            self,
            batch: tuple[Iterable, np.ndarray]) -> tuple[np.ndarray, float]:
        """
        Calculate the gradients of the loss function with respect to
        the model weights.

        Parameters
        ----------
        batch : tuple of Iterable and numpy.ndarray
            Current batch. Contains an Iterable of diagrams in index 0,
            and the targets in index 1.

        Returns
        -------
        tuple of np.ndarray and float
            The model predictions and the calculated loss.
        """
        diagrams, y = batch

        grad = self.model.evaluate(self.grad_loss, diagrams, y)
        pred = self.model(diagrams)
        loss = self.loss_fn(pred, y)

        self.gradient += grad

        return pred, loss

    def step(self) -> None:
        """
        Perform optimisation step.
        """
        updates, self.optimizer_state = self.optimizer.update(self.gradient,
                                                          self.optimizer_state)
        self.model.weights = optax.apply_updates(self.model.weights, updates)
        self.current_sweep += 1
        self.zero_grad()

    def state_dict(self) -> dict:
        """Return optimizer states as dictionary.

        Returns
        -------
        dict
            A dictionary containing the current state of the optimizer.

        """
        warnings.warn('OptaxOptimizer.state_dict not yet fully implemented')
        return {'current_sweep': self.current_sweep}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state of the optimizer from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing a snapshot of the optimizer state.

        """
        warnings.warn('OptaxOptimizer.load_state_dict'
                      'not yet fully implemented')
        self.current_sweep = state_dict['current_sweep']

    @staticmethod
    def get(optimizer : Callable):
        """
        Function that takes a callable returning an optax optimizer
        (i.e. an `optax.base.GradientTransformation` object) and returns
        a function that creates an OptaxOptimizer with that optimizer.

        Needed so that the class can be used with the current
        `QuantumTrainer` interface.
        """
        def create(model, hyperparams, loss):
            return OptaxOptimizer(model, optimizer, loss, hyperparams)
        return create
