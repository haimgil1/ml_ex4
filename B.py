import torch.nn as nn
import torch.nn.functional as F


class SecondNet(nn.Module):
    def __init__(self, image_size=28 * 28):
        super(SecondNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fd1 = nn.Dropout(p=0.1)
        self.fd2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))  # 28x28 -> 100
        x = self.fd1(x)
        x = F.relu(self.fc1(x))  # 100 -> 50
        x = self.fd2(x)
        # return F.log_softmax(x)
        return F.log_softmax(x, dim=1)
