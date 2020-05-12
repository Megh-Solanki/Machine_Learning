import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# importing data file for our model
path = r'D:\iNeuron ML Challenge\Bundle_2\houseRent\housing_train.csv'
data = pd.read_csv(path)

y = data.price  # target column

# features used for prediction
features = ['sqfeet', 'beds', 'baths', 'comes_furnished']
X = data[features]

# creating our model
rent_model = DecisionTreeRegressor(random_state=1)  # define
rent_model.fit(X, y)

preds = rent_model.predict(X)
print("Predictions are: ", rent_model.predict(X.head()))
print("Original values are: ", y.head().tolist())
