import numpy
import pandas
import torch
import torch.nn
import torch.optim
import torch.utils


def preprocess_train_data(data: pandas.DataFrame):
    target = data["label"].to_numpy()

    images = []
    for item in data.drop("label", axis=1).iterrows():
        images.append(item[-1].to_numpy().reshape(1, 28, 28).astype(numpy.float32))
    images = numpy.array(images)
    images = torch.from_numpy(images)

    data = []
    for (image, label) in zip(images, target):
        data.append((image, label))

    return data


def preprocess_test_data(data: pandas.DataFrame):
    images = []
    for item in data.iterrows():
        images.append(item[-1].to_numpy().reshape(1, 28, 28).astype(numpy.float32))
    images = numpy.array(images)
    images = torch.from_numpy(images)

    return images


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self,).__init__()
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, stride=2)

        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)

        self.fc1 = torch.nn.Linear(32 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(train_data):
    available_gpu = torch.cuda.is_available()
    if (available_gpu):
        gpu_count = torch.cuda.device_count()
        print(f"{gpu_count} gpus were detected.")
    else:
        print("no gpus were detected.")
        exit()

    device = torch.device("cuda:0")
    model = Model()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

    # data loader
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

    for epoch in range(1):
        for (images, labels) in train_data_loader:
            labels = labels.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss}")

    return model


def main():
    train_data = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")

    train_data = preprocess_train_data(train_data)
    test_images = preprocess_test_data(test)

    model = train(train_data)

    device = torch.device("cuda:0")
    test_images = test_images.to(device)
    result = torch.argmax(model(test_images), dim=1)
    result = result.to("cpu").detach().numpy()

    result = pandas.DataFrame(
        result, pandas.Series(numpy.arange(1, result.shape[0] + 1), name="ImageId"), columns=["Label"])
    result.to_csv("result.csv")


if __name__ == "__main__":
    main()
