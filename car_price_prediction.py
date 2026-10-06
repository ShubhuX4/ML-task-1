import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



# 1. Read the training data

train = pd.read_csv("train (1).csv")

print("Training data shape:", train.shape)
print(train.head())



# 2. Clean price and mileage


# Price looks like "$28,000", so we remove $ and comma
train["price"] = train["price"].str.replace("$", "", regex=False)
train["price"] = train["price"].str.replace(",", "", regex=False)
train["price"] = train["price"].astype(float)

# Mileage looks like "50,992 mi."
train["milage"] = train["milage"].str.replace(",", "", regex=False)
train["milage"] = train["milage"].str.replace(" mi.", "", regex=False)
train["milage"] = train["milage"].astype(float)




# 3. Separate X and y



# id is not really useful for predicting price
# clean_title has only one value in this dataset, so I removed it too

X = train.drop(["price", "id", "clean_title"], axis=1)
y = train["price"]



# 4. Handle missing values


# For text columns, use "Unknown"
text_columns = X.select_dtypes(include=["object"]).columns

for col in text_columns:
    X[col] = X[col].fillna("Unknown")


# Convert text columns into numbers
X = pd.get_dummies(X, columns=text_columns)


# Fill missing numerical values with the median
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)




# 5. Split the data



X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)




# 6. Train the model



model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)




# 7. Check the model



valid_prediction = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_prediction)
rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))

print("\nModel results")
print("Mean Absolute Error:", round(mae, 2))
print("RMSE:", round(rmse, 2))



# 8. Train again on all data


model.fit(X, y)



# 9. Read test data and predict


try:
    test = pd.read_csv("test.csv")

    test_ids = test["id"]

    # Do the same cleaning on test data
    test["milage"] = test["milage"].str.replace(",", "", regex=False)
    test["milage"] = test["milage"].str.replace(" mi.", "", regex=False)
    test["milage"] = test["milage"].astype(float)

    test_X = test.drop(["id", "clean_title"], axis=1)

    for col in text_columns:
        if col in test_X.columns:
            test_X[col] = test_X[col].fillna("Unknown")

    test_X = pd.get_dummies(test_X, columns=text_columns)

    # Make test columns exactly the same as training columns
    test_X = test_X.reindex(columns=X.columns, fill_value=0)

    test_X = test_X.fillna(X.median(numeric_only=True))
    test_X = test_X.fillna(0)

    prediction = model.predict(test_X)

    submission = pd.DataFrame({
        "id": test_ids,
        "price": prediction
    })

    submission.to_csv("submission.csv", index=False)

    print("\nDone!")
    print("submission.csv has been created.")
    print(submission.head())

except FileNotFoundError:
    print("\nCould not find test.csv.")
    print("Put test.csv in the same folder and run the program again.")
