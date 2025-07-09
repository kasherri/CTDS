import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def plot_latents_by_region(
    states: jnp.ndarray,
    list_of_dimensions: jnp.ndarray,
    save_path: str = None
):
    """
    Plot latent trajectories separately for each region.

    Args:
        states: (T, D) array of latent states
        list_of_dimensions: (R, C) array of latent dimensions per region and cell type
        save_path: optional path to save the plot
    """
    T, D = states.shape
    region_boundaries = jnp.cumsum(jnp.sum(list_of_dimensions, axis=1))
    region_boundaries = jnp.concatenate([jnp.array([0]), region_boundaries])

    num_regions = list_of_dimensions.shape[0]
    plt.figure(figsize=(4 * num_regions, 3))

    for r in range(num_regions):
        start, end = int(region_boundaries[r]), int(region_boundaries[r + 1])
        plt.subplot(1, num_regions, r + 1)
        for i in range(start, end):
            plt.plot(states[:, i], color='k', alpha=0.5)
        plt.title(f"True Latents - Region {r + 1}")
        plt.xlabel("Time")
        plt.yticks([0], ["0"])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()