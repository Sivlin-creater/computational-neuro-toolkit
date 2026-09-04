import numpy as np
from neuro_toolkit.spike_pipeline import LIFNeuron


def test_zero_current_produces_no_spikes():
    neuron = LIFNeuron()
    current = np.zeros(1000)
    v, spikes = neuron.simulate(current)
    assert len(spikes) == 0
    assert np.allclose(v, neuron.v_rest)


def test_high_current_produces_spikes():
    neuron = LIFNeuron()
    current = np.full(1000, 50.0)
    _, spikes = neuron.simulate(current)
    assert len(spikes) > 0
