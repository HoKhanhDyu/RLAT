from architech.vgg import VGG
from architech.cifar_10 import ResNet26
import torch
from dataset import get_num_classes
import torch.nn as nn
import torch.nn.functional as F

def get_architecture(arch: str, dataset: str, pytorch_pretrained: bool=False) -> torch.nn.Module:
    num_classes = get_num_classes(dataset)
    if arch == "vgg13":
        return VGG('VGG13', num_classes)
    elif arch == "vgg16":
        return VGG('VGG16', num_classes)
    elif arch == "vgg19":
        return VGG('VGG19', num_classes)
    
    elif arch == "restnet26":
        print("using resnet")
        if dataset == "cifar10":
            return ResNet26(in_channels=3, out_channels=num_classes)
        if dataset == "mnist":
            return ResNet26(in_channels=1, out_channels=num_classes)
        raise ValueError("Architecture not supported")
        
    else:
        raise ValueError("Architecture not supported")
    
def get_extractor(arch: str, dataset: str, pytorch_pretrained: bool=False) -> torch.nn.Module:
    if arch == "vgg13":
        return torch.nn.Sequential(*list(get_architecture(arch, dataset, pytorch_pretrained).features.children())[:-1])
    elif arch == "vgg16":
        return torch.nn.Sequential(*list(get_architecture(arch, dataset, pytorch_pretrained).features.children())[:-1])
    elif arch == "vgg19":
        return torch.nn.Sequential(*list(get_architecture(arch, dataset, pytorch_pretrained).features.children())[:-1])
    elif arch == "resnet26":
        return torch.nn.Sequential(*list(get_architecture(arch, dataset, pytorch_pretrained).children())[:-1])
    else:
        raise ValueError("Architecture not supported")
    
class MNIST_CC(nn.Module):
    def __init__(self):
        super(MNIST_CC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 28*28, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 512, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 2048, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 4096, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 256, out_features=10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        # out = F.softmax(out, dim=1)
        return out
    
class CONV_MNIST(nn.Module):
    def __init__(self):
        super(CONV_MNIST, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(12*12*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        out = self.classifier(x)
        # out = F.softmax(out, dim=1)
        return out

    
class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= input_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 512, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 128, out_features= output_size)
        )
    def forward(self, x):
        return self.classifier(x)
    
class DQN_Conv(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN_Conv, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear( in_features= 9216, out_features=1024),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear( in_features= 1024, out_features= 128),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear( in_features= 128, out_features= output_size)
        )
    def forward(self, x):
        return self.classifier(x)


