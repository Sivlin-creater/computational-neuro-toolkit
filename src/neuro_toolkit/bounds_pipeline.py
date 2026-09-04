import matplotlib.pyplot as plt
import numpy as np


def compute_hoeffding_bound(sample_sizes, epsilon):
    return 2.0 * np.exp(-2 * sample_sizes * (epsilon**2))


def run_bounds_experiment(
    num_trials=1500, max_n=400, step=20, epsilon=0.08, save_path=None
):
    sample_sizes = np.arange(10, max_n, step)
    empirical_violations = []
    p = 0.5  # Bernoulli parameter

    for n in sample_sizes:
        samples = np.random.binomial(n=1, p=p, size=(num_trials, n))
        means = samples.mean(axis=1)
        violations = np.mean(np.abs(means - p) >= epsilon)
        empirical_violations.append(violations)

    theoretical_bounds = [
        min(1.0, b) for b in compute_hoeffding_bound(sample_sizes, epsilon)
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(
        sample_sizes,
        empirical_violations,
        label=r"Empirical $P(|\bar{X}_n - \mu| \geq \epsilon)$",
        color="darkred",
    )
    plt.plot(
        sample_sizes,
        theoretical_bounds,
        "--",
        label="Theoretical Hoeffding Bound",
        color="black",
    )
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Probability")
    plt.title(rf"Concentration Bound Verification ($\epsilon={epsilon}$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

    return sample_sizes, empirical_violations, theoretical_bounds


if __name__ == "__main__":
    run_bounds_experiment(save_path="figures/statistical_bounds.png")
    print("Project 3: Figure saved to figures/statistical_bounds.png")
