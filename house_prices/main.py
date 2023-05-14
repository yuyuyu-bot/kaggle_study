import pandas
import sklearn.tree
import sklearn.metrics


def one_hot_encoding(data: pandas.DataFrame, label: str):
    data = pandas.concat([data, pandas.get_dummies(data[label], prefix=label)], axis=1)
    data = data.drop(columns=[label])
    return data


def preprocess_data(data: pandas.DataFrame, is_test_data=False):
    data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())

    data = one_hot_encoding(data, "MSZoning")
    data = one_hot_encoding(data, "Street")

    data["Alley"] = data["Alley"].fillna("No")
    data = one_hot_encoding(data, "Alley")

    data = one_hot_encoding(data, "LotShape")
    data = one_hot_encoding(data, "LandContour")
    data = one_hot_encoding(data, "Utilities")
    data = one_hot_encoding(data, "LotConfig")
    data = one_hot_encoding(data, "LandSlope")
    data = one_hot_encoding(data, "Neighborhood")
    data = one_hot_encoding(data, "Condition1")
    data = one_hot_encoding(data, "Condition2")
    data = one_hot_encoding(data, "BldgType")
    data = one_hot_encoding(data, "HouseStyle")
    data = one_hot_encoding(data, "RoofStyle")
    data = one_hot_encoding(data, "RoofMatl")
    data = one_hot_encoding(data, "Exterior1st")
    data = one_hot_encoding(data, "Exterior2nd")

    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data = one_hot_encoding(data, "MasVnrType")

    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

    data = one_hot_encoding(data, "ExterQual")
    data = one_hot_encoding(data, "ExterCond")
    data = one_hot_encoding(data, "Foundation")
    data = one_hot_encoding(data, "BsmtQual")
    data = one_hot_encoding(data, "BsmtCond")
    data = one_hot_encoding(data, "BsmtExposure")
    data = one_hot_encoding(data, "BsmtFinType1")
    data = one_hot_encoding(data, "BsmtFinType2")
    data = one_hot_encoding(data, "Heating")
    data = one_hot_encoding(data, "HeatingQC")
    data = one_hot_encoding(data, "CentralAir")
    data = one_hot_encoding(data, "Electrical")
    data = one_hot_encoding(data, "KitchenQual")
    data = one_hot_encoding(data, "Functional")

    data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
    data = one_hot_encoding(data, "FireplaceQu")

    data["GarageType"] = data["GarageType"].fillna("None")
    data = one_hot_encoding(data, "GarageType")

    data = data.drop("GarageYrBlt", axis=1)
    data = data.drop("GarageFinish", axis=1)

    data["GarageQual"] = data["GarageQual"].fillna(data["GarageCond"].mode())
    data = one_hot_encoding(data, "GarageQual")

    data["GarageCond"] = data["GarageCond"].fillna(data["GarageCond"].mode())
    data = one_hot_encoding(data, "GarageCond")

    data = one_hot_encoding(data, "PavedDrive")

    data = data.drop("PoolQC", axis=1)
    data = data.drop("Fence", axis=1)
    data = data.drop("MiscFeature", axis=1)

    data = one_hot_encoding(data, "SaleType")
    data = one_hot_encoding(data, "SaleCondition")

    # drop columns doesn't exist in test data
    if (not is_test_data):
        data = data.drop("Condition2_RRAe", axis=1)
        data = data.drop("Condition2_RRAn", axis=1)
        data = data.drop("Condition2_RRNn", axis=1)
        data = data.drop("Electrical_Mix", axis=1)
        data = data.drop("Exterior1st_ImStucc", axis=1)
        data = data.drop("Exterior1st_Stone", axis=1)
        data = data.drop("Exterior2nd_Other", axis=1)
        data = data.drop("Heating_Floor", axis=1)
        data = data.drop("Heating_OthW", axis=1)
        data = data.drop("GarageQual_Ex", axis=1)
        data = data.drop("HouseStyle_2.5Fin", axis=1)
        data = data.drop("RoofMatl_ClyTile", axis=1)
        data = data.drop("RoofMatl_Membran", axis=1)
        data = data.drop("RoofMatl_Metal", axis=1)
        data = data.drop("RoofMatl_Roll", axis=1)
        data = data.drop("Utilities_NoSeWa", axis=1)

    # drop columns containing na in test data
    data = data.drop("BsmtFinSF1", axis=1)
    data = data.drop("BsmtFinSF2", axis=1)
    data = data.drop("BsmtUnfSF", axis=1)
    data = data.drop("TotalBsmtSF", axis=1)
    data = data.drop("BsmtFullBath", axis=1)
    data = data.drop("BsmtHalfBath", axis=1)
    data = data.drop("GarageCars", axis=1)
    data = data.drop("GarageArea", axis=1)

    return data


def main():
    train = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")

    train = preprocess_data(train)
    test = preprocess_data(test, True)

    train_features = train[train.columns[train.columns != "SalePrice"]]
    train_target = train["SalePrice"]

    model = sklearn.tree.ExtraTreeRegressor()
    model.fit(train_features, train_target)
    train_predicted = model.predict(train_features)
    print(sklearn.metrics.accuracy_score(train_target, train_predicted))

    test_predicted = model.predict(test)
    result = pandas.DataFrame(test_predicted, test["Id"], columns=["SalePrice"])
    result.to_csv("result.csv", index_label=["Id"])


if __name__ == "__main__":
    main()
