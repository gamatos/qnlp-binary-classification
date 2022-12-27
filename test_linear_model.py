import jax
import numpy

from linear_model import LinearModel
from ansatz import (_hardware_efficient_ansatz,
                                _linear_IQP_ansatz_2q,
                                _trivial_combine,
                                _zero_ket)
from linear_model import build_linear_circuit


def test_weights():
    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    s, wc, cc = _linear_IQP_ansatz_2q()

    test_model = LinearModel.from_tokenised_sentences(
                                                    test_sentences,
                                                    s, wc, cc, 3, 1,
                                                    normalise=True)

    test_model.initialise_weights()

    # Test whether the correct number of weights have been generated
    assert test_model.word_weights.shape[0] == len(
        {s for ts in test_sentences for s in ts}) + 1

    padded_weights = test_model._padded_weights()

    # Test whether weight padding is being done correctly
    assert(numpy.isclose(padded_weights[1:, :-2],
                         test_model.word_weights)).all()

    new_sentences = [['Alice', 'likes', 'Bob'],
                     ['Bob', 'likes', 'the', 'park']]
    i = test_model._indices_from_diagrams(new_sentences)

    d = test_model.word_dict

    # Test whether weight indices are being correctly extracted from
    # sentences
    assert len(i) == len({s for ts in new_sentences for s in ts})
    assert (i == sorted({d['Alice'], d['likes'], d['the'],
                         d['park'], d['Bob']})).all()

    bi = test_model._batched_weight_indices(new_sentences)
    bw = test_model._indices_to_angles(bi)

    max_words = max(len(s) for s in test_sentences)

    # Test whether batched weights have the correct nr of sentences,
    # words and angles
    assert bw.shape == (len(new_sentences), max_words, 5)

    # Test whether the same words have the same weights
    assert numpy.isclose(bw[0, 2, :], bw[1, 0, :]).all()
    assert numpy.isclose(bw[0, 1, :], bw[1, 1, :]).all()

    padding_1 = max_words - len(new_sentences[0])
    padding_2 = max_words - len(new_sentences[1])

    # Test whether weight padding is being performed correctly
    assert numpy.isclose(bw[0, -padding_1:, :],
                         numpy.zeros((padding_1, 5))).all()
    assert numpy.isclose(bw[1, -padding_2:, :],
                         numpy.zeros((padding_2, 5))).all()

    assert numpy.isclose(bw[0, :-padding_1, -1],
                         numpy.ones(len(new_sentences[0]))).all()
    assert numpy.isclose(bw[1, :-padding_2:, -1],
                         numpy.ones(len(new_sentences[1]))).all()


def test_mask():
    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    s, wc, cc = _linear_IQP_ansatz_2q()

    test_model = LinearModel.from_tokenised_sentences(
                                                    test_sentences,
                                                    s, wc, cc, 3, 1,
                                                    normalise=True)

    test_model.initialise_weights()

    new_sentences = [['Alice', 'likes', 'Bob'],
                     ['Bob', 'likes', 'the', 'park']]

    word_mask = test_model._relevant_word_parameter_mask(new_sentences)
    mask = test_model._relevant_parameter_mask(new_sentences)

    i = test_model._indices_from_diagrams(new_sentences)

    # Test whether mask is correctly identifying weights relevant to
    # the words in the sentences
    complement = numpy.array([x for x in range(numpy.max(i)) if x not in i],
                             dtype=int)
    assert numpy.all(word_mask[i] == 0)
    assert numpy.all(word_mask[complement] == 1)
    assert numpy.all(mask[:word_mask.size] == word_mask.reshape(-1))
    assert numpy.all(mask[word_mask.size:] == 0)


def test_IQP_circuit():
    angles = jax.numpy.array([[0.687, 0.254, 0.555, 0.182, 1.],
                              [0.065, 0.54 , 0.129, 0.182, 1.],
                              [0.914, 0.287, 0.87 , 0.182, 1.],
                              [0.   , 0.   , 0.   , 0.   , 0.]])

    angles = angles.at[:, :-1].multiply(1/(2*jax.numpy.pi))

    result = numpy.array([-0.46324152-0.0329815j , -0.4305738 + 0.77390291j])

    s, wc, cc = _linear_IQP_ansatz_2q()
    c = build_linear_circuit(s, wc, cc, 3)

    assert numpy.isclose(c(angles), result).all()

    # Test whether initial combining circuit angle is being properly
    # ignored
    angles = angles.at[0, 3].set(100000)
    assert numpy.isclose(c(angles), result).all()

    """
    Value of result obtained by manually coding the following
    circuit in qiskit:
        ┌───────────┐┌───────────┐ ┌──────────┐┌───┐             ┌───┐»
   q_0: ┤ Rx(1.374) ├┤ Rz(0.508) ├─┤ Rx(1.11) ├┤ H ├──────■──────┤ H ├»
        └┬──────────┤└┬──────────┤┌┴──────────┤├───┤┌─────┴─────┐└───┘»
   q_1: ─┤ Rx(0.13) ├─┤ Rz(1.08) ├┤ Rx(0.258) ├┤ H ├┤ Rz(0.364) ├─|0>─»
         └──────────┘ └──────────┘└───────────┘└───┘└───────────┘     »
   c: 2/══════════════════════════════════════════════════════════════»
                                                                      »
   «
   «q_0: ─────────────────────────────────────────────────■───────────
   «     ┌───────────┐┌───────────┐┌──────────┐┌───┐┌─────┴─────┐
   «q_1: ┤ Rx(1.828) ├┤ Rz(0.574) ├┤ Rx(1.74) ├┤ H ├┤ Rz(0.364) ├─|0>─
   «     └───────────┘└───────────┘└──────────┘└───┘└───────────┘
   «c: 2/═════════════════════════════════════════════════════════════
   «
    """


def test_hardware_efficient_ansatz():
    n_qubits = 4
    hwa = _hardware_efficient_ansatz(n_qubits, 1)
    initial_state = _zero_ket(n_qubits).reshape((2,)*n_qubits)
    circuit = build_linear_circuit(initial_state, hwa, _trivial_combine,
                                   2*n_qubits)

    word_angles = ([[0.2836, 0.5755, 0.9429, 0.9413, 0.9613, 0.9321, 0.2447,
                     0.6542],
                    [0.3286, 0.8961, 0.6342, 0.0388, 0.1657, 0.8546, 0.5099,
                     0.8327]])

    word_angles = numpy.array(word_angles) / (2 * numpy.pi)
    word_angles = numpy.c_[word_angles, numpy.ones(2)]

    # Value of result obtained by manually coding the circuit in qiskit
    test_res = [-0.5337-0.2487j,  0.0468-0.0682j,  0.2153-0.0849j,
                -0.019-0.0352j, 0.1832-0.3163j,  0.1202+0.1065j,
                0.0848+0.0231j,  0.0331-0.0338j, 0.1351-0.2151j,
                -0.0076+0.0256j,  0.358+0.0985j, -0.0019-0.0276j,
                -0.0814-0.2989j,  0.0933+0.0266j, -0.0208-0.3249j,
                -0.0616-0.0537j]

    test_res = numpy.array(test_res)

    res = circuit(word_angles)

    test_res = numpy.round(test_res, 3)
    assert(numpy.allclose(numpy.round(res, 3).reshape(-1), test_res))


def test_sentence_evaluation():
    s, wc, cc = _linear_IQP_ansatz_2q()

    _f = build_linear_circuit(s, wc, cc, 3)

    def f(x):
        return LinearModel._normalise_vector(_f(x))

    vf = jax.vmap(f)

    test_sentences = [['Alice', 'likes', 'the', 'city', 'park'],
                      ['Bob', 'likes', 'Alice']]

    test_model = LinearModel.from_tokenised_sentences(
                                                    test_sentences,
                                                    s, wc, cc, 3, 1,
                                                    normalise=True)

    test_model.initialise_weights()

    d = test_model.word_dict

    new_sentence = [['Alice', 'likes', 'Bob']]
    new_sentence_indices = numpy.array([d['Alice']+1, d['likes']+1,
                                        d['Bob']+1])

    w = jax.numpy.array(test_model._padded_weights()[new_sentence_indices])

    bi = test_model._batched_weight_indices(new_sentence)
    bw = jax.numpy.array(test_model._indices_to_angles(bi))

    # Tests whether weight batching gives the same result as
    # manually constructed unbatched evaluation
    assert numpy.isclose(vf(bw)[0], f(w)).all()

    # Test whether model evaluation gives the same result as evaluation
    # directly on circuit
    assert numpy.isclose(vf(bw), test_model(new_sentence)).all()


def test_variation():
    # Tests whether, given some sentences, varying the model's weights
    # which are relevant to evaluating those sentences produces a change
    # in the output, and varying those which are not relevant produces
    # no change.

    test_sentences = [['Alice', 'likes', 'skating'],
                      ['Bob', 'admires', 'Alice']]

    s, wc, cc = _linear_IQP_ansatz_2q()

    test_model = LinearModel.from_tokenised_sentences(
                                                test_sentences,
                                                s, wc, cc, 3, 1,
                                                normalise=True)

    test_model.initialise_weights()

    new_sentences = [['Alice', 'admires', 'Bob'],
                     ['Bob', 'admires', 'skating']]

    y0 = test_model.predictions(new_sentences, False)
    x = test_model.weights

    mask = test_model._relevant_parameter_mask(new_sentences)
    mask_complement = [not x for x in mask]

    mask_individual = [i for i, x in enumerate(mask) if x == 0]
    mask_complement_individual = [i for i, x in enumerate(mask_complement)
                                  if x == 0]

    for i in mask_complement_individual:
        delta = numpy.array([a == i for a in range(x.size)])
        delta = numpy.ma.masked_array(delta, mask=mask)

        test_model.weights = x + 0.1*delta

        assert numpy.allclose(test_model.predictions(new_sentences, False),
                              y0)
        test_model.weights = x

    for i in mask_individual:
        delta = numpy.array([a == i for a in range(x.size)])
        delta = numpy.ma.masked_array(delta, mask=mask)

        test_model.weights = x + 0.1*delta

        assert not numpy.allclose(test_model.predictions(new_sentences,
                                                         False),
                                  y0)
        test_model.weights = x


def test_normalisation():
    # Test whether model is properly normalising states
    test_array = numpy.array([[   10,           1j*10],
                              [1j*10, numpy.sqrt(300)],
                              [   1.,              0.],
                              [   0.,              1.]])

    test_res = numpy.array([[1/2, 1/2],
                            [1/4, 3/4],
                            [ 1.,  0.],
                            [ 0.,  1.]])

    res = LinearModel._normalise_vector(test_array)

    assert numpy.allclose(res, test_res)


def test_padding_gradient():
    # Test whether padding of weights so that they can be batched
    # does not spoil the computation of the gradient

    angles = jax.numpy.array([[0.687, 0.254, 0.555, 0.182, 1.],
                              [0.065, 0.54 , 0.129, 0.182, 1.],
                              [0.914, 0.287, 0.87 , 0.182, 1.],
                              [0.   , 0.   , 0.   , 0.   , 0.]])

    s, wc, cc = _linear_IQP_ansatz_2q()
    c = build_linear_circuit(s, wc, cc, 3)

    def dummy_cost(x):
        return jax.numpy.linalg.norm(c(x))

    g = jax.grad(dummy_cost)

    assert numpy.allclose(g(angles)[:3, :], g(angles[:3, :]))
    assert not numpy.all(g(angles[3:, :]))


def test_gradient_evaluation():
    # Test whether gradient is being properly evaluated

    s, wc, cc = _linear_IQP_ansatz_2q()

    sentences = [['Bob', 'dances']]
    test_model = LinearModel.from_tokenised_sentences(
                                                sentences,
                                                s, wc, cc, 3, 1,
                                                normalise=True)
    test_model.initialise_weights()

    def loss(y, y0):
        return jax.numpy.linalg.norm(y-y0)

    def loss_from_weights(w, d, y):
        i = test_model._batched_weight_indices(d)
        x = test_model._indices_to_angles(i, w)
        y0 = test_model.circuit_evaluator(x, test_model.normalise)
        res = loss(y0, y)
        return res

    g_test = jax.grad(loss_from_weights)
    g = test_model.grad_loss(loss)

    y = jax.numpy.array([[1., 0.]])
    g_test_val = g_test(test_model.weights, sentences, y)
    g_val = test_model.evaluate(g, sentences, y)

    assert numpy.allclose(g_val, g_test_val)
