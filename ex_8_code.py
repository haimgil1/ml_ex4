import torch
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
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
        x = F.relu(self.fc0(x))  # 28x28 -> 100
        x = F.relu(self.fc1(x))  # 100 -> 50
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(x)
        return F.log_softmax(x, dim=1)


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


class FourthNet(nn.Module):
    def __init__(self, image_size=28 * 28):
        super(FourthNet, self).__init__()
        self.image_size = image_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def load_and_split_data(transforms):
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)

    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms)

    # We need to further split our training dataset into training and validation sets.

    # Define the indices
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = int(0.2 * len(train_dataset))  # define the split size

    # Define your batch_size
    batch_size = 64

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, validation_loader, test_loader


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def run(model, train_loader, validation_loader, test_loader, model_name, num_of_epoch=100):
    print("Running {}".format(model_name))
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    validation_loss_dict = {}
    validation_acc_dict = {}
    training_loss_dict = {}
    training_acc_dict = {}

    best = 0.0
    for epoch in range(1, num_of_epoch + 1):
        train_loader_size = len(train_loader.dataset)
        test_loader_size = len(test_loader.dataset)
        train_size = int(0.8 * train_loader_size)
        validation_size = int(0.2 * train_loader_size)

        train(model, train_loader, optimizer)
        loss, acc = test(model, train_loader, train_size, True, epoch, "Train")
        training_loss_dict[epoch] = loss
        training_acc_dict[epoch] = acc
        loss, acc = test(model, validation_loader, validation_size, True, epoch, "Validation")
        validation_loss_dict[epoch] = loss
        validation_acc_dict[epoch] = acc
        loss, acc = test(model, test_loader, test_loader_size, False, epoch, "Test")
        if acc > best:
            best = acc
            print(best)
            write_pred(model, test_loader)
            # label1, = plt.plot(training_loss_dict.keys(), training_loss_dict.values(), "b-", label="Training average loss per epoch")
            # label2, = plt.plot(validation_loss_dict.keys(), validation_loss_dict.values(), "r-", label="Validation average loss per epoch")
            # # label3, = plt.plot(training_acc_dict.keys(), training_acc_dict.values(), "g-", label="Training accurate per epoch")
            # # label4, = plt.plot(validation_acc_dict.keys(), validation_acc_dict.values(), "y-", label="Validation accurate per epoch")
            # plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
            # plt.legend(handler_map={label2: HandlerLine2D(numpoints=4)})
            # # plt.legend(handler_map={label3: HandlerLine2D(numpoints=4)})
            # # plt.legend(handler_map={label4: HandlerLine2D(numpoints=4)})
            #
            # plt.show()


def test(model, test_loader, size, is_train, epoch, name):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= size
    acc = 100.0 * correct / size
    if is_train:
        print('{} Epoch: {} [{}/{} ({:.2f}%)] Loss:{:.4f}'
              .format(name, epoch, correct, size, acc, test_loss))
    else:
        print('{} set: Average loss: {:.4f}, Accuracy: {}/{}({:.2f}%)\n'
              .format(name, test_loss, correct, size, acc))
    return test_loss, acc


def write_pred(model, test_loader):
    model.eval()
    with open('test.pred', 'w') as f:
        for data, label in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            preds = pred.numpy()

            for pred in preds:
                f.writelines(str(pred[0]) + '\n')


def main():
    train_loader, validation_loader, test_loader = load_and_split_data(transforms)
    # model = FirstNet()
    # run(model, train_loader, validation_loader, test_loader, "A")
    model = SecondNet()
    run(model, train_loader, validation_loader, test_loader, "b")
    model = ThirdNet()
    run(model, train_loader, validation_loader, test_loader, "C")
    model = FourthNet()
    run(model, train_loader, validation_loader, test_loader, "D")


if __name__ == '__main__':
    main()
