# define trained letters

classes = ("bet", "dalet", "shin")
qtd_classes = len(classes)

# creates the neural network class

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, qtd_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import numpy as np

# data_transform to normalize images

data_transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(image, model):
	img = PIL.ImageOps.invert(Image.fromarray(np.uint8(image*255)).convert('RGB'))
	image_tensor = data_transform(img).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	input = input.to(device)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index

def classify(image, model):
	return classes[predict_image(image, model)]