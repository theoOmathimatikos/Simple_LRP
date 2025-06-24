import os; from os.path import join
import matplotlib.pyplot as plt
import torch

def X_times_W_plus_b(X, W, b, add_bias=False):
    """
    Performs a basic computation, omnipresent across the calculation of lrp:
    multiplying weights and inputs and adding bias. 

    Args:
        X (torch.tensor): the layer's input
        W (torch.tensor): the layer's weights
        b (torch.tensor): the layer's bias
        add_bias (bool): True in order to add bias
    
    Returns:
        torch.tensor
    """

    if len(X.shape) == 1 or X.shape[0]==3:
        R = torch.matmul(torch.diag(X), W)
    else:
        R = torch.matmul(torch.diag_embed(X), W)

    if add_bias:
        R += b / X.shape[1]

    return R


def visualize(data, R_data):
    """A method for visualizing the calculated Relevances."""

    for i in range(R_data.shape[0]):

        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(data[i], cmap='gray')
        axs[1].imshow(R_data[i], cmap='coolwarm')

        for ax in axs:
            ax.axis('off')
        
        plt.tight_layout()

        plt.savefig(join(os.getcwd(), f"Results/{i}_rel.png"))
        plt.close()
