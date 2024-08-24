from torchvision import datasets
from torchvision import transforms

transforms_list = transforms.Compose([
transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))
])
data_train = datasets.CIFAR10(root='./data',train=True,download=True,transform=transforms_list)
data_test = datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms_list)

