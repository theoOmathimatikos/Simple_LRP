import os; from os.path import join
import torch; import torch.nn as nn


class simpleDNN4l(nn.Module):
    """
    A simple DNN with for linear layers, each followed by ReLU (except the last layer where
    logits are calculated with the use of Softmax).
    """

    def __init__(self, neurons_per_layer):
        """
        Args:
            neurons_per_layer ([ints]): A list of integers, each representing the 
                number of neurons of the corresponding layer. 
        """

        super().__init__()
        self.layers = neurons_per_layer
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model_shape = neurons_per_layer
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(neurons_per_layer[0], 
                      neurons_per_layer[1]), 
            nn.ReLU(),
            nn.LazyLinear(neurons_per_layer[2]), nn.ReLU(),
            nn.LazyLinear(neurons_per_layer[3]), nn.ReLU(),
            nn.LazyLinear(10), nn.Softmax()
        )

        self.forward(torch.rand((1, neurons_per_layer[0])))
        self._init_weights()

        self.net.to(self.device)


    def forward(self, X):
        return self.net(X)
    
    def _init_weights(self):
    
        for m in self.net:
            if isinstance(m, nn.Linear):
    
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)


def load_model(layers, ckpt_name):
    """
    Load model. At this point, use your own class of models and checkpoints.
    
    Args: 
        layers ([ints]): represents the number of neurons at each layer, 
        ckpt_name (str): name of the model saved
    
    Returns:
        torch.nn.Module
    """

    model = simpleDNN4l(layers)
    tot_pth = join(os.getcwd(), "Checkpoints", ckpt_name)

    ckpt = torch.load(tot_pth)
    model.load_state_dict(ckpt)

    return model