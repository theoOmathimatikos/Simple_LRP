import torch 
import torch.nn as nn
from LRP.utils import X_times_W_plus_b

class LRP:
    """
    This is the class implementing the LRP algorithm. Instead of adding hooks, this 
    algorithm performs a forward pass, gathering data at each layer to calculate the
    relevance.
    """

    def __init__(self, model, epsilon=1e-6):
        """Args:
            model (nn.Module): A FeedForward Neural Network (with linear layers and activation functions),
            epsilon (float): The epsilon parameter of the LRP method 
        """
        self.model = model
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epsilon = epsilon

    
    def initialize_R_tot(self, X, no_batch=False):
        """
        Initializes a tensor of suitable shape for calculating relevance scores.
        Contains only ones, for a neutral effect on the first multiplication in lrp
        calculation.
        
        Args:
            X (torch.tensor): the input tensor,
            no_batch (bool): True if X is one sample, False if it is a batch
        
        Returns:
            torch.tensor
        """
        
        if len(X.squeeze().shape)==2 and no_batch:
            R_tot = torch.eye(X.numel()).to(self.device)  
        else:
            R_tot = torch.eye(X[0, ::].numel()).repeat(X.shape[0], 1, 1).to(self.device)
        
        return R_tot


    def calculate_lrp(self, X, add_bias=False, relative_lrp=False):
        """
        How this method works is as follows: 

        Args:
            X (torch.tensor): a tensor of shape (1, h, w) or (n_batch, h, w),
            add_bias (bool): ,
            relative_lrp (bool): 
        
        Returns:
            torch.tensor of shape (1, h, w) or (n_batch, h, w)
        """

        R_tot  = self.initialize_R_tot(X)
        mods = [l for n, l in self.model.named_modules() if '.' in n]

        # squeeze channels if image is gray
        X = X.squeeze()

        if X.device.type == 'cpu':
            X = X.to(self.device)

        for i, layer in enumerate(mods):

            if isinstance(layer, nn.Linear):

                W = layer.weight.T
                b = layer.bias

                W, b = W.detach(), b.detach()

                # Perform diag(X)*W multiplication
                R = X_times_W_plus_b(X, W, b, add_bias)
                
                # Define normalizing term
                dm = 0 if len(R.size()) == 2 else 1
                Z = R.shape[1] if relative_lrp else torch.sum(R, dim=dm)
                # Add epsilon
                Z[torch.abs(Z)<1e-6] += self.epsilon
                if not relative_lrp and add_bias: Z += b

                # Normalize by the total
                if len(R.size()) > 2 and not relative_lrp:
                    Z = Z.unsqueeze(1)
                R /= Z

                if torch.sum(torch.isnan(R))>0:
                    print('nan values')

                R_tot = torch.matmul(R_tot, R)
            
            # elif isinstance(layer, nn.Sequential):
            #     continue

            X = layer(X)

        
        return R_tot, X


