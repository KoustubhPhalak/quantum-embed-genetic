'''Do it for first time only when downloading dataset'''
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to Tensor
])

# Load the MNIST training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data',  # Directory to save the data
    train=True,  # Load the training set
    transform=transform,  # Apply transformations
    download=True  # Download the dataset if it doesn't exist
)

# Load the MNIST training dataset
test_dataset = torchvision.datasets.MNIST(
    root='./mnist_data',  # Directory to save the data
    train=False,  # Load the training set
    transform=transform,  # Apply transformations
    download=True  # Download the dataset if it doesn't exist
)


