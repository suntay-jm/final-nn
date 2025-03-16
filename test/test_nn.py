import numpy as np
from nn import NeuralNetwork

# define a small test architecture
nn_arch = [
    {"input_dim": 4, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
]
nn = NeuralNetwork(nn_arch=nn_arch, lr=0.01, seed=42, batch_size=2, epochs=10, loss_function="binary_cross_entropy")

# small test dataset
X_train = np.random.randn(4, 5)  # (input_dim, num_samples)
y_train = np.random.randint(0, 2, (1, 5))  # binary labels


def test_initialization():
    """check if the neural network initializes parameters correctly"""
    assert len(nn.arch) == len(nn._param_dict) // 2, "architecture layers and parameter count not matching up"
    
    for i, layer in enumerate(nn.arch, start=1):
        W_shape = nn._param_dict[f"W{i}"].shape
        b_shape = nn._param_dict[f"b{i}"].shape
        assert W_shape == (layer["output_dim"], layer["input_dim"]), f"incorrect shape for weights - W{i}"
        assert b_shape == (layer["output_dim"], 1), f"incorrect shape for biases - b{i}"


def test_forward_pass():
    """ensure forward propagation produces correctly shaped output"""
    y_hat, _ = nn.forward(X_train)
    assert y_hat.shape == (1, 5), "forward pass output shape incorrect"


def test_loss_function():
    """verify binary cross-entropy loss computation"""
    y_hat = np.clip(np.random.rand(1, 5), 1e-7, 1 - 1e-7)  # avoid log(0)
    loss = nn._binary_cross_entropy(y_train, y_hat)
    assert np.isscalar(loss) and loss > 0, f"BCE loss should be a positive scalar, it is dtype{loss}"


def test_backpropagation():
    """ensure backpropagation correctly updates parameters"""
    initial_W1 = nn._param_dict["W1"].copy()
    y_hat, cache = nn.forward(X_train)
    grad_dict = nn.backprop(y_train, y_hat, cache)
    nn._update_params(grad_dict)
    updated_W1 = nn._param_dict["W1"]
    assert not np.allclose(initial_W1, updated_W1), "weights not updated after backpropagation"


def test_training_pipeline():
    """train for a few epochs and check if loss decreases"""
    train_loss, _ = nn.fit(X_train, y_train, X_train, y_train)
    assert train_loss[-1] < train_loss[0], "training loss did not decrease over epochs"