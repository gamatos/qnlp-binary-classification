"""
JAX Ansatze
==========
File containing circuit ansatze as JAX-compatible functions
"""
from __future__ import annotations

import jax
import numpy
import qujax


#################################

def _zero_ket(L : int):
    a = jax.numpy.zeros(2**L, dtype=complex)
    a = a.at[0].set(1)
    a = a.reshape((2,)*L)
    return a


_zk2 = _zero_ket(1).reshape(-1)

def _linear_IQP_ansatz_2q(initial_word_state=_zk2):
    """
    Returns a function that takes a set of angles and evaluates
    the `IQPAnsatz` with one qubit and one layer

    The angles should be supplied as a `jax.numpy.ndarray`.
    Padding is represented as a row having all zeros.
    """
    import jax.numpy as jnp

    H = 1/jnp.sqrt(2)*jnp.array([[1, 1], [1, -1]])
    HH = jnp.kron(H, H)
    Id_local = jnp.eye(2)
    proj = jnp.kron(jnp.array([[1., 0.], [0., 0.]]), Id_local)
    Id = jnp.eye(4)

    def _crz(t):
        t = t * (2*numpy.pi)
        return jnp.diag(jnp.array([1,  jnp.exp(-1j * t),
                                   1,  jnp.exp(1j * t)]))

    def _rz(t):
        t = t * (2*numpy.pi)
        return jnp.diag(jnp.array([jnp.exp(-1j*t), jnp.exp(1j*t)]))

    def _rx(t):
        t = t * (2*numpy.pi)
        return jnp.array([[jnp.cos(t), -1j*jnp.sin(t)],
                          [-1j*jnp.sin(t), jnp.cos(t)]])

    def _IQP_2q_word_circuit(left, angles):
        x = angles
        res = _rx(x[2]) @ _rz(x[1]) @ _rx(x[0]) @ initial_word_state
        return res

    def _IQP_2q_combining_circuit(left, right, angles):
        # Flag indicating whether current word is padding
        x = angles

        res = jnp.kron(right, left)
        res = HH @ res
        res = _crz(x[3]) @ res
        res = proj @ res
        res *= 1/jnp.linalg.norm(res)
        return res.reshape(2, 2)[0, :]

    return initial_word_state, _IQP_2q_word_circuit, _IQP_2q_combining_circuit


def _trivial_combine(left, right, angles):
    return right


def _hardware_efficient_ansatz(n_qubits, layers):
    circuit_gates, circuit_qubit_inds, circuit_params_inds = [], [], []

    circuit_gates += ['Ry'] * n_qubits
    circuit_qubit_inds += [[n_qubits - 1 - i] for i in range(n_qubits)]
    circuit_params_inds += [[i] for i in range(n_qubits)]

    circuit_qubit_inds += [[n_qubits - 1 - i] for i in range(n_qubits)]
    circuit_params_inds += [[i + len(circuit_gates)] for i in range(n_qubits)]
    circuit_gates += ['Rz'] * n_qubits

    circuit_gates += ['CX'] * (n_qubits - 1)
    circuit_qubit_inds += [[n_qubits - 1 - i, n_qubits - 2 - i]
                           for i in range(n_qubits - 1)]
    circuit_params_inds += [[]] * (n_qubits - 1)

    param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                       circuit_qubit_inds,
                                                       circuit_params_inds)

    def _param_to_st_scan(statetensor_in, params):
        return param_to_st(params, statetensor_in), None

    def _hwa(left, angles):
        angles = angles.reshape(layers, -1)
        angles = 2 * angles
        res, _ = jax.lax.scan(_param_to_st_scan, left, angles)
        return res

    return _hwa


def _multi_cnot_and_measure(n_qubits):
    circuit_gates = ['CX'] * n_qubits
    circuit_qubit_inds = [[n_qubits - i, 0] for i in range(n_qubits)]
    circuit_params_inds = [[]] * n_qubits

    param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                       circuit_qubit_inds,
                                                       circuit_params_inds)

    def _mcn(left):
        left = jax.numpy.stack((left, jax.numpy.zeros_like(left)))
        left = param_to_st(left)
        probabilities = left.reshape(2, -1)
        probabilities = jax.numpy.square(jax.numpy.abs(probabilities))
        probabilities = probabilities.sum(axis=1)
        return probabilities

    return _mcn

def _reduced_hea(n_qubits, layers):
    circuit_gates, circuit_qubit_inds, circuit_params_inds = [], [], []

    circuit_qubit_inds += [[n_qubits - 1 - i] for i in range(n_qubits)]
    circuit_params_inds += [[i + len(circuit_gates)] for i in range(n_qubits)]
    circuit_gates += ['Rz'] * n_qubits

    circuit_gates += ['CX'] * (n_qubits - 1)
    circuit_qubit_inds += [[n_qubits - 1 - i, n_qubits - 2 - i]
                           for i in range(n_qubits - 1)]
    circuit_params_inds += [[]] * (n_qubits - 1)

    param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                       circuit_qubit_inds,
                                                       circuit_params_inds)

    def _param_to_st_scan(statetensor_in, params):
        return param_to_st(params, statetensor_in), None

    def _hwa(left, angles):
        angles = angles.reshape(layers, -1)
        angles = 2 * angles
        res, _ = jax.lax.scan(_param_to_st_scan, left, angles)
        return res

    return _hwa

#Untested
def _qaoa(n_qubits, layers):
    circuit_gates, circuit_qubit_inds, circuit_params_inds = [], [], []

    circuit_gates += ['Rxx'] * (n_qubits - 1)
    circuit_qubit_inds += [[n_qubits - 1 - i, n_qubits - 2 - i]
                           for i in range(n_qubits - 1)]
    circuit_params_inds += [[i] for i in range(n_qubits)]

    circuit_qubit_inds += [[n_qubits - 1 - i] for i in range(n_qubits)]
    circuit_params_inds += [[i + len(circuit_gates)] for i in range(n_qubits)]
    circuit_gates += ['Rz'] * n_qubits

    param_to_st = qujax.get_params_to_statetensor_func(circuit_gates,
                                                       circuit_qubit_inds,
                                                       circuit_params_inds)

    def _param_to_st_scan(statetensor_in, params):
        return param_to_st(params, statetensor_in), None

    def _qaoa_circuit(left, angles):
        angles = angles.reshape(layers, -1)
        angles = 2 * angles
        res, _ = jax.lax.scan(_param_to_st_scan, left, angles)
        return res

    return _qaoa_circuit
