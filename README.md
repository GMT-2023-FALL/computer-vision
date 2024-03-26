# Environment Setup
```python
conda create -n pytorch python=3.8
conda activate pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Models
## Baseline Model
- ### Convolutional Layers:
* * conv1: The first convolutional layer accepts a single-channel (e.g., grayscale) picture input with a padding of two to preserve spatial dimensions after convolution. It generates six feature maps by using six 5x5 filters (or kernels). Padding is employed to keep spatial dimensions the same as the original LeNet-5, which was built for 32x32 pixel inputs, although it has most likely been updated for 28x28 pixel inputs (as shown in datasets such as MNIST).
* * conv2: The second convolutional layer takes the six feature maps from conv1 and applies 16 filters of size 5x5, without padding, to minimize the spatial dimensions of the resultant feature maps.

- ### Pooling Layers: 
Every convolutional layer is followed by a 2x2 max pooling operation (F.max_pool2d). This cuts the feature maps' spatial dimensions in half, making the representation more compact and allowing higher-level characteristics to emerge.
- ### Fully Connected Layers:
* * fc1: The first fully connected layer flattens the output from the preceding max pooling layer and connects it to a layer with 120 nodes.
* * fc2: The second fully connected layer takes the 120 nodes as input and connects them to 84 nodes.
* * fc3: The final fully connected layer maps the 84 nodes to 10 output nodes, corresponding to the number of classes in the classification task.

- ### Weight Initialization:
Weights are initialized using the Kaiming/He method, which is suitable for layers followed by ReLU activations. This method helps prevent the vanishing or exploding gradients problem in deep neural networks.

- ### Activation Function:
The ReLU (Rectified Linear Unit) activation function is used throughout the network. It introduces non-linearity to the model, allowing it to learn more complex functions and generally works well in deep learning models.

- ### Loss Function:
The `nn.CrossEntropyLoss()` from PyTorch is used, which combines a softmax layer with the cross-entropy loss in one single class. This loss function is appropriate for multi-class classification problems.

- ### Optimizer:
The Adam optimizer (`torch.optim.Adam`) is utilized for adjusting network weights. Adam is known for its efficiency with large datasets and high-dimensional spaces.

- ### Learning Rate:
A learning rate of 0.001 is used, which determines the step size at each iteration while moving toward a minimum of the loss function. This value is a commonly used default for the Adam optimizer.

- ### Batch Size:
The batch size set as `32` within the DataLoader. It defines the number of samples that will be propagated through the network before the optimizer updates the model parameters.

- ### Epochs:
Training is set to run for `10` epochs, which means the learning algorithm will work through the entire training dataset a total of 10 times.


![Baseline_Train](Baseline%20Model_metrics.png)
![cm](Confusion%20Matrix%20for%20Baseline%20Model.png)






## Variant Model 1
The key changes between the version 1 and baseline models are the number of output channels in the first convolutional layer and the layout of the second convolutional layer.
<br/>
Because in baseline model, it is clear showing that the model is overfitting, so we can try to reduce the number of output channels in the first convolutional layer to see if it helps to reduce overfitting.
Reducing the number of output channels can help simplify the model by reducing the number of model parameters. Such a modify could be done to find out the influence of the model's simplicity on performance or to reduce overfitting.
![Variant1](Variant%20Model%201_metrics.png)
![cm1](Confusion%20Matrix%20for%20Variant%20Model%201.png)
- Result Discussion:
## Variant Model 2
But for variant 1, the model is still overfitted, so we thought of adding dropout to the fully-connected layer using the dropout rate of 0.5 to see if it helps to reduce overfitting.
![Variant1](Variant%20Model%202_metrics.png)
![cm1](Confusion%20Matrix%20for%20Variant%20Model%202.png)
- Result Discussion:
## Variant Model 3
After seeing that Model 2 did not overfit, we replaced ReLU with LeakyReLU in this variant in all the places where we used the activation function.The parameter negative_slope=0.01 of LeakyReLU defines the slope of the function in the negative portion of the function, which allows for the gradient to be non-zero in negative intervals, which helps to prevent neuron death problems .
![Variant1](Variant%20Model%203_metrics.png)
![cm1](Confusion%20Matrix%20for%20Variant%20Model%203.png)
- Result Discussion:
## Variant Model 4
Showing signs of slow convergence during model training in model three, our reduced learning rate allows for finer weight updates, which helps the model find the minimum of the loss function more quickly. We can try to set the learning rate as half of the previous `0.0005` to see if it helps to speed up the convergence of the model.
![Variant1](Variant%20Model%204_metrics.png)
![cm1](Confusion%20Matrix%20for%20Variant%20Model%204.png)
- Result Discussion:

# Top-1 Accuracy
| Model | Training Top-1 Accuracy | Validation Top-1 Accuracy |
|-------|-------------------------|---------------------------|
| Base  | 93.51%                  | 89.92%                  |
| Var1  | 93.03%                  | 89.05%                    |
| Var2  | 86.68%                  | 88.06%                    |
| Var3  | 88.11%                  | 89.61%                    |
| Var4  | 86.31%                  | 87.91%                    |

# Pair Comparison
## Base vs Var1
- Base model has a higher training and validation accuracy than Var1 model.
## Var1 vs Var2
## Var2 vs Var3
## Var3 vs Var4

# Choice Task
