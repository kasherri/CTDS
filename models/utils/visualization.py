import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_and_save_parameters(A, C, Q, R, save_folder, J=None):
    """ plot and save the parameters of the simulation """
    mc = matplotlib.colors
    # plot and save all the matrices in one figure
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    vmin = np.min(A)
    vmax = np.max(A)
    color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = vmin)
    plt.imshow(A, norm=color_norm, cmap="bwr")
    # plt.imshow(A, cmap='bwr', aspect='auto', vcenter=0)
    plt.xticks([])
    plt.yticks([])
    plt.title('A', fontsize=20)
    cbar = plt.colorbar()
    # cbar.set_ticks([-1.0,0,1.0])
    # cbar.set_ticklabels(['-1.0','0','1.0'])
    plt.subplot(2, 2, 2)
    vmax = np.max(C)
    color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = -1)
    plt.imshow(C, cmap='bwr', norm=color_norm)
    plt.title('C', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    # cbar.set_ticks([-1.0,0,1.0])
    # cbar.set_ticklabels(['-1.0','0','1.0'])
    plt.subplot(2, 2, 3)
    plt.imshow(Q, cmap='bwr')
    plt.title('Q', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(R, cmap='bwr')
    plt.title('R', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()

    if save_folder is not None:
        plt.savefig(save_folder+'model_params.png', bbox_inches='tight', dpi=300)
    if J is not None:
        plt.figure(figsize=(6, 4))
        vmin = np.min(J)
        vmax = np.max(J)
        color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = vmin)
        plt.imshow(J, cmap='bwr', norm=color_norm)
        plt.title('J', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()
        if save_folder is not None:
            plt.savefig(save_folder[:-4] + '_J.png', bbox_inches='tight', dpi=300)
        plt.show()
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