import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image
import glob
import keyboard
import random

cifar10_map = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}


model_location = "Model\\model1.pth"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Cuda version {torch.cuda_version} enabled.")

transform_list = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# Hyperparameters
batch_size=32
learning_rate=0.00003
epochs=200


# Cifar10 is 32X32 Color Images
train_data = datasets.CIFAR10(root="./CIFAR10", transform=transform_list, train=True, download=True)
test_data = datasets.CIFAR10(root="./CIFAR10", transform=transform_list, train=False, download=True)

train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
#         self.MP2D = nn.MaxPool2d(2) #Kernel size = 2x2
#         self.lin1 = nn.Linear(6272, 1024)
#         self.lin2 = nn.Linear(1024, 128)
#         self.lin3 = nn.Linear(128, 10)
#         self.dropoutLin = nn.Dropout(0.5)
#         self.dropoutConv = nn.Dropout(0.8)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         # x = self.dropoutConv(x)
#         x = F.relu(self.conv2(x))
#         # x = self.dropoutConv(x)
#         x = F.relu(self.conv3(x))
#         x = self.dropoutConv(x)
#         x = self.MP2D(x)
#         # print(x.shape)
#         x = x.view(-1, 6272)
#         x = F.relu(self.lin1(x))
#         x = self.dropoutLin(x)
#         x = F.relu(self.lin2(x))
#         x = self.dropoutLin(x)
#         x = F.relu(self.lin3(x))
#         # x = self.dropoutLin(x)
#         # return F.softmax(x, dim=1)
#         return x


##############################################################################################################




######################################################################################################################
model = torchvision.models.resnet50(pretrained=False).to(device)
# model = CNN().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Trainable Parameters = {pytorch_total_params}")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train():
    model.train()
    for epoch in range(epochs):
        model.train()
        for batch_index, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            predicted = model(data)
            loss = loss_function(predicted, label)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if batch_index%50 == 0:
                print(f"On Batch {batch_index}/{len(train_data)/batch_size}, with loss {loss.item()}")
        print(f"Epoch = {epoch}")
        model.eval()
        test()
    print("ran")
    torch.save(model, model_location)

def test():
    model.eval()
    len_test_data = len(test_data)
    num_correct = 0

    for batch_index, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)

        predicted = model(data)
        # print(predicted.shape)
        maxes = torch.argmax(predicted, dim=1)

        num_correct += (maxes == label).sum()

    print(f"Num Correct = {num_correct}")
    print(f"Num Total = {len_test_data}")
    print(f"Percent Correct = {(num_correct/len_test_data)*100}")
    model.train()

def load_single():
    capture_location = "custom.jpg"
    transform_load = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model2 = torch.load(model_location)
    model2 = model2.to(device)
    model2.eval()
    while True:
        while True:
            if keyboard.is_pressed("l"):
                break;
        image_custom = cv2.imread(capture_location)
        image_custom = cv2.resize(image_custom, (32, 32))
        image_custom = transform_load(image_custom)
        with torch.no_grad():
            single_trial_tensor = image_custom.unsqueeze(0)
            single_trial_tensor = single_trial_tensor.to(device)
            test_individual(model2, single_trial_tensor)

            #Display
            img_show = Image.open(capture_location)
            img_show.show()


def test_individual(model, tensor):
    model.eval()
    with torch.no_grad():
        category_guessed = cifar10_map[model(tensor).max(1)[1].item()]
        print(f"Guess = {category_guessed}")
    model.train()

for epoch in range(200):
    train()
    test()
    scheduler.step()

# model = torch.load(model_location).to(device)
# test()
# load_single()


# fig = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols*rows+1):
#     sample_idx=random.randint(0, len(train_data))
#     img, label = train_data[sample_idx]
#     print(img.size())
#     print(img.squeeze().size())
#     fig.add_subplot(rows, cols, i)
#     plt.axis("off")
#     img = transforms.Grayscale()(img)
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()