import torch.nn as nn
import torch.nn.functional as F


class FirstNet(nn.Module):
    def __init__(self, image_size=28 * 28):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x)) # 28x28 -> 100
        x = F.relu(self.fc1(x)) # 100 -> 50
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(x)
        return F.log_softmax(x,dim=1)
