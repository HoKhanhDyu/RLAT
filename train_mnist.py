import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='datasets/dataset_cache', train=True, download=False, transform=transform)
test_dataset = MNIST(root='datasets/dataset_cache', train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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
        out = F.softmax(out, dim=1)
        return out

model = MNIST_CC()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
    model.train()
    return num_correct/num_samples

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # print(images.shape)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Train Accuracy: {accuracy(train_loader, model):.4f}, Test Accuracy: {accuracy(test_loader, model):.4f}')

torch.save(model.state_dict(), 'trained_model/mnist_cc.pth')

print("Training completed.")
