import torch.nn as nn
import torch.nn.functional as F


class ThirdNet(nn.Module):
    def __init__(self, image_size=28 * 28):
        super(ThirdNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = F.relu(self.bn1(x))  # 28x28 -> 100
        x = self.fc1(x)
        x = F.relu(self.bn2(x))  # 100 -> 50
        # return F.log_softmax(x)
        return F.log_softmax(x, dim=1)
