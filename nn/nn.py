# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        
        # computes linear transformation
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        """
        applies weight matrix (W_curr) to previous activations (A_prev) and adds the bias term (b_curr)
        """
        
        if activation == "relu":
            A_curr = np.maximum(0, Z_curr) # replaces negative values with 0 and keeps positive values unchanged 
        
        elif activation == "sigmoid":
            A_curr = 1 / (1 + np.exp(-Z_curr))
        
        """
        squashes values between 0-1
        """
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        A_curr = X # setting initial to input
        cache = {} # empty dictionary to store intermediate values 
        
        for idx, layer in enumerate(self.arch):
            """
            for each layer in self.arch, extract weight and biases 
            then call _single_forward() to compute weighted sum before activation (Z_curr) and and activated output (A_curr)
            """
            layer_idx = idx + 1 # adjusts layer index to start from 1 instead of 0 (since weight and bias keys start from 1 not 0)

            # getting layer's weights and biases 
            W_curr = self._param_dict[f"W{layer_idx}"]
            b_curr = self._param_dict[f"b{layer_idx}"]

            # applying linear transformation and applying activation
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, layer["activation"]) # W_curr @ A_prev _b_curr

            # storing activation and pre-activation values in cache for backprop
            cache[f"A{layer_idx}"] = A_curr
            cache[f"Z{layer_idx}"] = Z_curr

            
        return A_curr, cache # A_curr has final predictions and cache contains all intermediate activation for back prop

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        
        m = A_prev.shape[1] # extracting batch size to normalize gradients across all samples in the batch
        
        if activation_curr == "relu": # computes dZ_curr for ReLU activation
            dZ_curr = dA_curr * (Z_curr > 0)
        elif activation_curr == "sigmoid": # computes dZ_curr for sigmoid activation
            sig = 1 / (1 + np.exp(-Z_curr)) # derivative of sigmoid 
            dZ_curr = dA_curr * sig * (1 - sig) # backpropagates gradients through sigmoid
        
        dW_curr = np.dot(dZ_curr, A_prev.T) / m # computes weight gradient
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m # computes bias gradient 
        
        dA_prev = np.dot(W_curr.T, dZ_curr) # computes how much the previous layer contributed to the error and backprops the error to earlier layers
        
        return dA_prev, dW_curr, db_curr # passing gradients to optimizer 

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {} # empty dictionary to store gradients 
        
        m = y.shape[1] # extracting batch size from ground truth labels 

        dA_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
        """
        computing gradient of the loss function
        using BCE loss derivative formula 
        first gradient used in backprop
        """
        
        for idx, layer in reversed(list(enumerate(self.arch))):# looping backwards through the network
            layer_idx = idx + 1
            """
            gets the weight and bias and forward pass data 
            pass into current gradient 
            pass the activation function for proper differentiation
            then compute weight, bias, and activation gradients 
            """
            dA_prev, dW_curr, db_curr = self._single_backprop(
                self._param_dict[f"W{layer_idx}"], self._param_dict[f"b{layer_idx}"],
                cache[f"Z{layer_idx}"], cache[f"A{layer_idx - 1}"] if layer_idx > 1 else y_hat,
                dA_prev, layer["activation"]
            )

            # stores computed gradients in grad_dict
            grad_dict[f"dW{layer_idx}"] = dW_curr
            grad_dict[f"db{layer_idx}"] = db_curr
            
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx in range(len(self.arch)):
            layer_idx = idx + 1
            """
            loop through all layers and extract weight and bias gradients 
            then apply gradient descent update 
            and adjust weights and biases based on gradient descent 
            """
            self._param_dict[f"W{layer_idx}"] -= self._lr * grad_dict[f"dW{layer_idx}"]
            self._param_dict[f"b{layer_idx}"] -= self._lr * grad_dict[f"db{layer_idx}"]


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]: 
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """


        train_loss, val_loss = [], [] # initializing empty lists to track loss 
        
        for _ in range(self._epochs):
            y_hat, cache = self.forward(X_train) # passing training data to get predictions and intermediate and pre-activations 
            grad_dict = self.backprop(y_train, y_hat, cache) # compute gradients using backprop 
            """
            computes gradients for weights and biases using back prop 
            compares predictions with true labels
            """
            self._update_params(grad_dict) # applies gradient descent updates using computed gradients

            # computing and storing loss for training/validation set 
            if self._loss_func == "binary_cross_entropy":
                train_loss.append(self._binary_cross_entropy(y_train, y_hat))
                val_loss.append(self._binary_cross_entropy(y_val, self.forward(X_val)[0]))
            elif self._loss_func == "mean_squared_error":
                train_loss.append(self._mean_squared_error(y_train, y_hat))
                val_loss.append(self._mean_squared_error(y_val, self.forward(X_val)[0]))
            else:
                raise ValueError(f"Unsupported loss function: {self._loss_func}")

        return train_loss, val_loss # returns lists of loss values per epoch 


    def predict(self, X: ArrayLike) -> ArrayLike:
            """
            This function returns the prediction of the neural network.

            Args:
                X: ArrayLike
                    Input data for prediction.

            Returns:
                y_hat: ArrayLike
                    Prediction from the model.
            """
            return (self.forward(X)[0] > 0.5).astype(int) # apply threshold at .5 for classification

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z)) # squash values between 0-1

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig = self._sigmoid(Z) # compute sigmoid 
        return dA * sig * (1 - sig) # apply (sigmoid) derivative formula 

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z) # set negative values to 0, keep positives unchanged 

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z > 0) # gradient is 1 for Z > 0, otherwise 0 

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        m = y.shape[1] # number of samples
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m # BCE formula


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
            """
            Binary cross entropy loss function derivative for backprop.

            Args:
                y_hat: ArrayLike
                    Predicted output.
                y: ArrayLike
                    Ground truth output.

            Returns:
                dA: ArrayLike
                    partial derivative of loss with respect to A matrix.
            """
            return -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)) # BCE gradient formula

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean((y - y_hat) ** 2) # average squared difference between y and y_hat 

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return -2 * (y - y_hat) / y.shape[1] # gradient of MSE with respect to predictions 