# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        #original nn.Module is an alternative when the data flow isn't a straight line: skip connections, multiple inputs or outputs, or branching.
        #nn.ModuleList or ModuleDict if connection is not defined and its not registered (no show in .parameters(), ignored in .to(device))
        #other alternatives:
        #Functional style.  torch.nn.functional (F.relu, F.conv2d, and so on) inside forward for operations that have no parameters.
        #nn.Sequential with an OrderedDict. This is still Sequential, but it lets you name the layers
        # they all sit in pytorch/torch_venv/lib/python3.12/site-packages/torch/nn/modules/container.py
        # interestingly, nn has a .pop() method
        '''
        pop(self, key: int | slice) -> Module:
        """
        Pop ``key`` from self.
        """
        v = self[key]
        del self[key]'''
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss() #cool docs in pytorch/torch_venv/lib/python3.12/site-packages/torch/nn/modules/loss.py
# be carefull if targets are probabilities not indices! manual softmax aplication may be needed. values between ``[0,1]`` and the total distribution summing to 1 required, will fail silently if not!
"""PyTorch does not validate whether the values provided in ``target`` lie in the range ``[0,1]``
        or whether the distribution of each data sample sums to ``1``.
        No warning will be raised and it is the user's responsibility
        Target can be indices or probabilities.
   The `input` is expected to contain the unnormalized logits for each class (which do `not` need
       to be positive or sum to 1, in general).
        
        
        def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    """
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    '''

┌───────────────────────┬──────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
│       Mechanism       │             Acts on              │               Lasts               │                              Effect                              │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ requires_grad_(False) │ a tensor or parameter (the leaf) │ permanent, until you flip it back │ No gradient is computed for that tensor.                         │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ detach()              │ one tensor, producing a new one  │ that new tensor only              │ Cuts the graph at that point. The original is unchanged.         │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ no_grad()             │ a block of code                  │ inside the block                  │ Nothing computed there is recorded.                              │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ inference_mode()      │ a block of code                  │ inside the block                  │ Same as no_grad, but faster and stricter.                        │
├───────────────────────┼──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ model.eval()          │ the model's layers               │ until .train()                    │ Not about gradients. It changes dropout and batchnorm behaviour. │
└───────────────────────┴──────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────────────────────────────┘

    pytorch/torch_venv/lib/python3.12/site-packages/torch/autograd/grad_mode.py 

    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "set_multithreading_enabled",
    "enforce_grad_layout_policy"

    no_grad:
    model.eval() is not an alternative. It switches dropout and batchnorm behaviour but doesn't stop gradient tracking, so you normally use it together with no_grad or inference_mode.
    alternative: .detach(): This cuts a single tensor out of the graph, which is useful for logging or for using a value as a constant. For example, loss.detach() or x.detach().cpu().
    itorch.inference_mode()
    inference_mode is faster, but its output tensors can never be used in autograd later. no_grad is slightly slower, and its outputs can still be used in training afterwards

    detach() gives you a copy of a tensor that has forgotten how it was computed, requires_grad=False,. It shares the same data, but it's no longer connected to the graph.
    Without detach(), one of two things happens, depending on the case.
        Logging losses: Stored tensors keep their graph alive. The leak is small when you call backward() each step and large when you don't.
        GAN-style cases: gradients flow into places you didn't intend. For example, the discriminator's loss also computes gradients for the generator, which wastes compute and can silently corrupt the training. You wouldn't get an error, just worse results, or a "backward through the graph a second time" error if the graph was already freed.


    "Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.'''
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')