## Simple and Efficient Implementation of LRP

[LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) stands as one of the most recognized and well-accepted methods in XAI. At its core, what LRP does is to propagate the output score back to the input through the layers, by considering how the layers' parameters affect it at each step. This is captured by the rules

$$R_{i, j}^{l, l+1} = a_{i, j}* R_j^{l+1},$$

where 

$$a_{i, j} = \frac{z_i*w_{ij}}{\sum_i z_i*w_{ij}}$$

and

$$R_i^l = \sum_j R_{i, j}^{l, l+1}.$$

Thus, by selecting of the last layer $O$, $R^O = f(x)$, where $f$ is our model and $x$ being the input data, this calculated score is being calculated in a backward pass. 

Yet, its [official implementation](https://github.com/chr5tphr/zennit) might be somewhat strange, applying hooks and replacing gradients with the rules of the LRP method applied (i.e. $\epsilon$-LRP, $\alpha\beta$-LRP etc.). 

<!-- Also, the relevance values of the input and intermediate layers do not sum up to 1 (check [here](https://github.com/chr5tphr/zennit/issues/213)) -->

If you notice carefully, for simple FeedForward Neural Networks (with linear layers and activation functions) LRP can be calculated in a simple forward pass: we simply need to calculate the values $a_{i, j}$ and leave the later value $R_j^{l+1}$ for later. These values are being multiplied to a matrix of suitable shape, gathering all information.

If you are interested in the basics ($\epsilon$-rule) or want to test Relative-LRP (where the divisor now is been eliminated), you can easily try the code.

## How to use

Download torch and run code with `python run.py`. To use a different model, consider adding its architecture in `LRP.model.py` and its parameters inside `Checkpoints`. To use a different dataset, consider passing your dataloader inside the `configuration` method in `run.py`.

The default model is a simple DNN with 4 layers and ReLU activations, and the default data is a subset of [Typeface MNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist). Enjoy! 