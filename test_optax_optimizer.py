from typing import Callable

import jax
import optax

from lambeq.training import Dataset, QuantumTrainer
from optax_optimizer import OptaxOptimizer


class DummyModel():
    def __init__(self) -> None:
        super().__init__()
        self.initialise_weights()

    def __call__(self, *args, **kwds):
        return self.predictions(*args, **kwds)

    def from_checkpoint():
        pass

    def symbols():
        return ['a', 'b', 'c']

    def initialise_weights(self):
        self.weights = jax.numpy.array([1., 2., 3.])

    def predictions(self, x, w=None):
        x = jax.numpy.array(x)
        if w is None:
            w = self.weights
        jax.debug.print("w {}", w)
        jax.debug.print("x {}", x)
        return w * x

    def loss(self, loss_fn):
        def loss_from_weights(w, x, y):
            y0 = self.predictions(x, w)
            jax.debug.print("y {} {}, {}", y, type(y), type(y[0]))
            jax.debug.print("y0 {}", y0)
            res = loss_fn(y0, y[0])
            jax.debug.print("res {}", res)
            return res
        return loss_from_weights

    def grad_loss(self, loss_fn):
        return jax.grad(self.loss(loss_fn))

    def evaluate(
                self,
                f: Callable,
                diagrams,
                y: jax.numpy.ndarray) -> jax.numpy.ndarray:
        res = f(self.weights, diagrams, y)
        return res


def test_optax_optimizer():
    m = DummyModel()

    def loss_fn(y0, y):
        return jax.numpy.linalg.norm(y0 - y)

    o = OptaxOptimizer(m, optax.adam, loss_fn, {'learning_rate': 5e-2})

    d = jax.numpy.array([[4., 2., 1.]])
    y = jax.numpy.array([[1., 1., 1.]])

    for _ in range(150):
        o.backward((d, y))
        o.step()

    assert jax.numpy.allclose(m.weights, y/d, atol=1e-1)


def test_optax_trainer():
    m = DummyModel()
    SEED = 2
    BATCH_SIZE = 1
    EPOCHS = 150

    def loss_fn(y0, y):
        return jax.numpy.linalg.norm(y0 - y)

    trainer = QuantumTrainer(m,
                             loss_function=loss_fn,
                             epochs=EPOCHS,
                             optimizer=OptaxOptimizer.get(optax.adam),
                             optim_hyperparams={'learning_rate': 5e-2},
                             evaluate_on_train=True,
                             verbose='text',
                             seed=SEED)

    d = jax.numpy.array([[4., 2., 1.]])
    y = jax.numpy.array([[1., 1., 1.]])

    train_dataset = Dataset(d, y, batch_size=BATCH_SIZE, backend=jax.numpy)
    val_dataset = Dataset(2*d, 2*y, backend=jax.numpy)

    trainer.fit(train_dataset, val_dataset)

    assert jax.numpy.allclose(m.weights, y/d, atol=1e-1)
