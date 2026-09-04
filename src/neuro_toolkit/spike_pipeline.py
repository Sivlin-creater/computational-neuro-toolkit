import numpy as np
import matplotlib.pyplot as plt


class LIFNeuron:
    def __init__(self, v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0, tau_m=10.0, r=1.0, dt=0.1):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_m = tau_m
        self.r = r
        self.dt = dt

    def simulate(self, current_trace):
        steps = len(current_trace)
        v = np.zeros(steps)
        v[0] = self.v_rest
        spikes = []

        for t in range(1, steps):
            dv = (-(v[t-1] - self.v_rest) + self.r *
                  current_trace[t]) * (self.dt / self.tau_m)
            v[t] = v[t-1] + dv
            if v[t] >= self.v_thresh:
                v[t] = self.v_reset
                spikes.append(t * self.dt)
        return v, np.array(spikes)


def plot_raster_and_psth(spike_trains, duration_ms, bin_size_ms=5.0, save_path=None):
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for trial_idx, spikes in enumerate(spike_trains):
        ax_raster.vlines(spikes, trial_idx + 0.5, trial_idx +
                         1.5, color="black", linewidth=0.8)
    ax_raster.set_ylabel("Trial")
    ax_raster.set_title("LIF Population Spike Response")

    bins = np.arange(0, duration_ms + bin_size_ms, bin_size_ms)
    all_spikes = np.concatenate(spike_trains) if len(
        spike_trains) > 0 else np.array([])
    counts, _ = np.histogram(all_spikes, bins=bins)
    firing_rate = counts / (len(spike_trains) * (bin_size_ms / 1000.0))

    ax_psth.bar(bins[:-1], firing_rate, width=bin_size_ms,
                align="edge", color="steelblue")
    ax_psth.set_xlabel("Time (ms)")
    ax_psth.set_ylabel("Firing Rate (Hz)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    neuron = LIFNeuron()
    duration = 500.0  # ms
    steps = int(duration / neuron.dt)

    trials = 20
    spike_trains = []
    for _ in range(trials):
        noisy_input = np.random.normal(loc=18.0, scale=4.0, size=steps)
        _, spikes = neuron.simulate(noisy_input)
        spike_trains.append(spikes)

    plot_raster_and_psth(spike_trains, duration,
                         save_path="figures/raster_psth.png")
    print("Project 1: Figure saved to figures/raster_psth.png")
