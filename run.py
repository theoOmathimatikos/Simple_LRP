import os; from os.path import join
import torch

from LRP.model import load_model
from LRP.lrp import LRP
from LRP.utils import visualize


def run_lrp(model, data, labels, add_bias=False, rel_lrp=False):
    """
    Performs an efficient version of lrp, without the use of hooks are gradient replacements. 
    Only performs the original lrp (with the choice of using bias or not) or relative lrp (where
    the divisor is the shape of the layer).

    It saves the results as images in folder `Results`.

    Args:
        model (torch.nn.Module): the model to perform lrp,
        data (torch.tensor): the data (should be in shape (1, h, w) or (n_batch, h, w)),
        add_bias (bool): True if we want to add bias at the divisor of the original lrp, 
        rel_lrp (bool): True to calculate relative lrp
    
    """

    # LRP
    lrp_class = LRP(model)
    lrp_data, _ = lrp_class.calculate_lrp(data, add_bias, rel_lrp)

    # Get scores only for class of interest (matching label)
    lrp_data = lrp_data[torch.arange(lrp_data.shape[0]), :, labels]

    # Detach and transform data
    data = data.view(data.shape[0], 28, 28)
    lrp_data = lrp_data.detach().cpu().view(lrp_data.shape[0], 28, 28)

    # Visualization
    visualize(data, lrp_data)


def configuration(model_name, data_pth):

    model = load_model([784, 320, 120, 40], model_name)
    data = torch.load(join(os.getcwd(), data_pth))

    samples, labels = data['samples'], data['labels']
    run_lrp(model, samples, labels)
    

if __name__=="__main__":

    model_name = "DNN_TMNIST.pt"
    data_pth = "data.pt"

    configuration(model_name, data_pth)