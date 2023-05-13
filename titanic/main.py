import pandas
import sklearn.tree
import sklearn.metrics


def preprocess_data(data):
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


def main():
    train = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")

    train = preprocess_data(train)
    test = preprocess_data(test)

    feature_labels = ["Pclass", "Sex", "Age", "Embarked_C", "Embarked_Q", "Embarked_S"]
    train_features = train[feature_labels]
    train_target = train["Survived"]

    model = sklearn.tree.ExtraTreeClassifier()
    model.fit(train_features, train_target)
    train_predicted = model.predict(train_features)
    print(sklearn.metrics.accuracy_score(train_target, train_predicted))

    test_features = test[feature_labels]
    test_predicted = model.predict(test_features)
    result = pandas.DataFrame(test_predicted, test["PassengerId"], columns=["Survived"])
    result.to_csv("result.csv", index_label=["PassengerId"])


if __name__ == "__main__":
    main()
