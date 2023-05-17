import numpy
import pandas
import sklearn.metrics
import torch
import torch.nn
import torch.optim
import torch.utils


def preprocess_data(data: pandas.DataFrame):
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data = data.dropna(subset=["Embarked"])
    data = data.drop("Cabin", axis=1)
    data = data.drop("Fare", axis=1)
    data = data.drop("Ticket", axis=1)

    # one-hot encoding "Embarked"
    data = pandas.concat([data, pandas.get_dummies(data["Embarked"], prefix="Embarked")], axis=1)
    # drop original "Embarked"
    data = data.drop(columns=["Embarked"])

    # one-hot encoding "Sex"
    data["Sex"] = pandas.get_dummies(data["Sex"], drop_first=True)

    return data


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(6, 12)
        self.fc2 = torch.nn.Linear(12, 6)
        self.fc3 = torch.nn.Linear(6, 6)
        self.fc4 = torch.nn.Linear(6, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def main():
    train = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")

    train = preprocess_data(train)
    test = preprocess_data(test)

    feature_labels = ["Pclass", "Sex", "Age", "Embarked_C", "Embarked_Q", "Embarked_S"]
    train_features = train[feature_labels]
    train_target = train["Survived"]

    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)

    print(train_features)
    train_data = []
    for (data, label) in zip(train_features.to_numpy(), train_target):
        train_data.append([torch.from_numpy(data.astype(numpy.float32)), label])

    # data loader
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)

    for epoch in range(100000):
        for (data, labels) in train_data_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss}")

    result = model(torch.from_numpy(train_features.to_numpy().astype(numpy.float32)))
    result = torch.argmax(result, dim=1)
    result = result.detach().numpy()
    print(sklearn.metrics.accuracy_score(train_target, result))

    test_features = test[feature_labels]
    result = model(torch.from_numpy(test_features.to_numpy().astype(numpy.float32)))
    result = torch.argmax(result, dim=1)
    result = result.detach().numpy()
    result = pandas.DataFrame(result, test["PassengerId"], columns=["Survived"])
    result.to_csv("result.csv", index_label=["PassengerId"])


if __name__ == "__main__":
    main()
