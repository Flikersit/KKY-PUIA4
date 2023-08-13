import numpy as np
from numpy import load
import sklearn
import sklearn.metrics
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import requests
from requests.auth import HTTPDigestAuth
from urllib.request import urlretrieve
import os


def onehot(sequence):
    input_hot = []
    residua = ["A",
              "R",
              "N",
              "D",
              "C",
              "Q",
              "E",
              "G",
              "H",
              "I",
              "L",
              "K",
              "M",
              "F",
              "P",
              "S",
              "T",
              "W",
              "Y",
              "V",]
    sequence = list(sequence)
    for i in range(len(sequence)):
        if(sequence[i] in residua):
            ind = residua.index(sequence[i])
        else:
            ind = -1
        hot = []
        for k in range(20):
            if(ind == k):
                hot.append(1)
            else:
                hot.append(0)
        input_hot.append(hot)
    return input_hot 




def out(cislo):
    if cislo == "1":
        return [0, 1]
    else:
        return [1, 0]


# def one_hot_back(vector, max_length):
#     y = []
#     for i in range(len(vector)):
#         y.append(vector[len(vector) - i - 1])
#     if (len(y) > max_length):
#         result = []
#         for i in range(max_length):
#             result.append(y[i])
#         y = result
#     while (len(y) < max_length):
#         y.insert(0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     return y


def supplement(vector, max_length):
    if (len(vector) > max_length):
        result = []
        for i in range(max_length):
            result.append(vector[i])
        vector = result
    while (len(vector) < max_length):
        vector.insert(0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return vector


def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url, file)
        print("File downloaded")


class ProteinLSTM(torch.nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, num_layers, batch_size):
        super(ProteinLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.forward_lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim * 2, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Forward LSTM
        forward_output, _ = self.forward_lstm(x)
        forward_output = forward_output[:, -1, :]
        elmo_embeddings = self.linear(forward_output)
        elmo_embeddings = self.softmax(elmo_embeddings)

        return elmo_embeddings

class Soltest(Dataset):

    def __init__(self, max_length):
        download("https://github.com/Flikersit/kky-puia4/blob/main/test_input.txt", "test_input.txt")
        download("https://github.com/Flikersit/kky-puia4/blob/main/test_output.txt", "test_output.txt")
        test_input = open("test_input.txt")
        test_output = open("test_output.txt")
        testinputtensor = []
        # testinputtensory = []
        for line in test_input:
            x = onehot(line)
            # y = one_hot_back(x, max_length)
            x = supplement(x, max_length)
            testinputtensor.append(x)
            # testinputtensory.append(y)
        self.testinputtensor = torch.tensor(testinputtensor, dtype=torch.float)
        # self.testinputtensory = torch.tensor(testinputtensory, dtype=torch.long)
        output = []
        for line1 in test_output:
            output.append(out(line1))
        self.output = torch.tensor(output, dtype=torch.float)

    def __getitem__(self, index):
        return self.testinputtensor[index, :, :], self.output[index, :]

    def __len__(self):
        return self.testinputtensor.size(dim=0)


class Soltrain(Dataset):

    def __init__(self, max_length):
        download("https://github.com/Flikersit/kky-puia4/blob/main/train_input.txt", "train_input.txt")
        download("https://github.com/Flikersit/kky-puia4/blob/main/train_output.txt", "train_output.txt")
        train_input = open("train_input.txt")
        train_output = open("train_output.txt")
        traininputtensor = []
        # traininputtensory = []
        for line in train_input:
            x = onehot(line)
            # y = one_hot_back(x, max_length)
            x = supplement(x, max_length)
            traininputtensor.append(x)
            # traininputtensory.append(y)
        self.traininputtensor = torch.tensor(traininputtensor, dtype=torch.float)
        # self.traininputtensory = torch.tensor(traininputtensory, dtype=torch.long)
        output = []
        for line1 in train_output:
            output.append(out(line1))
        self.output = torch.tensor(output, dtype=torch.float)

    def __getitem__(self, index):
        return self.traininputtensor[index, :, :], self.output[index, :]

    def __len__(self):
        return self.traininputtensor.size(dim=0)


class Solval(Dataset):

    def __init__(self, max_length):
        download("https://github.com/Flikersit/kky-puia4/blob/main/val_input.txt", "val_input.txt")
        download("https://github.com/Flikersit/kky-puia4/blob/main/val_output.txt", "val_output.txt")
        val_input = open("val_input.txt")
        val_output = open("val_output.txt")
        valinputtensor = []
        # valinputtensory = []
        for line in val_input:
            x = onehot(line)
            # y = one_hot_back(x, max_length)
            x = supplement(x, max_length)
            valinputtensor.append(x)
            # valinputtensory.append(y)
        self.valinputtensor = torch.tensor(valinputtensor, dtype=torch.float)
        # self.valinputtensory = torch.tensor(valinputtensory, dtype=torch.long)
        output = []
        for line1 in val_output:
            output.append(out(line1))
        self.output = torch.tensor(output, dtype=torch.float)

    def __getitem__(self, index):
        return self.valinputtensor[index, :, :], self.output[index, :]

    def __len__(self):
        return self.valinputtensor.size(dim=0)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataloader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)


def main():
    max_length = 125
    num_epochs = 100
    rate_learning = 1e-4
    device = get_default_device()
    dataset = Soltrain(max_length)
    dataset1 = Solval(max_length)
    dataset2 = Soltest(max_length)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    dataloader1 = DataLoader(dataset=dataset1, batch_size=16, shuffle=True)
    dataloader2 = DataLoader(dataset=dataset2, batch_size=1, shuffle=True)
    dataloader = DeviceDataloader(dataloader, device)
    dataloader1 = DeviceDataloader(dataloader1, device)
    dataloader2 = DeviceDataloader(dataloader2, device)
    model = ProteinLSTM(20, 1024, 125, 16)
    to_device(model, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=rate_learning)
    history = []
    history1 = []
    accuracy_train = []
    accuracy_val = []
    accuracy_test = []
    for epochs in range(num_epochs):
        running_loss = 0
        val_loss = 0
        model.train()
        for i, data in enumerate(dataloader):
            # Every data instance is an input + label pair
            inputs1, labels = data
            # Zero your gradients for every batch!
            optim.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs1.float())
            # Accuracy

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optim.step()

            # Gather data and report
            running_loss += loss.item()
            labels1 = np.argmax(labels, axis=1)
            outputs1 = np.argmax(outputs.detach().numpy(), axis=1)
            accuracy_train.append(accuracy_score(labels1, outputs1))

        history.append(running_loss)
        model.eval()
        for k, data in enumerate(dataloader1):
            with torch.no_grad():
                inputs1, labels = data
                outputs = model(inputs1.float())
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                labels1 = np.argmax(labels, axis=1)
                outputs1 = np.argmax(outputs.detach().numpy(), axis=1)
                accuracy_val.append(accuracy_score(labels1, outputs1))
        history1.append(val_loss)
    correct = 0
    total = 0
    for j, data in enumerate(dataloader2):
        with torch.no_grad():
            inputs1, labels = data
            outputs = model(inputs1.float())
            if outputs.detach().numpy() == labels:
                correct += 1
            total += 1
    print("Total accurancy ", correct / total)
    print("History", history)
    print("History1", history1)
    print("Accuracy train", accuracy_train)
    print("accuracy_val", accuracy_val)
    #plt.plot(history)
    #plt.xlabel("epochs")
    #plt.ylabel("loss")
    #plt.title("Train dataset")
    #plt.show()
    #plt.plot(history1)
    #plt.xlabel("epochs")
    #plt.ylabel("loss")
    #plt.title("Validation dataset")
    #plt.show()
    #plt.plot(accuracy_train)
    #plt.show()


main()
#print(onehot("AJSKFNKGLNAFLKNVNFLKBLKSLKGNBLK"))
