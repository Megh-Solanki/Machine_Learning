# author: Meghrajsinh Solanki

# importing necessary libraries and modules
import pandas as pd
from numpy import nan
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

# importing data file for our model
path = r'D:\iNeuron ML Challenge\Bundle_2\houseRent\housing_train.csv'
data = pd.read_csv(path)
y = data.price  # target column


# print(data['laundry_options'].describe()) --> this code is to describe
# laundry_options column. So that we can have the value with most occurences
# in the column and we can replace nan/ empty values with that value
data['laundry_options'] = data['laundry_options'].replace(nan, 'w/d in unit')
# print(data['laundry_options'].isnull().sum()) --> this code is just to verify
# that we don't have any nan/ empty values in our column after the replacement


# print(data['parking_options'].describe()) --> same as laundry_options column
data['parking_options'] = data['parking_options'].replace(nan, 'off-street parking')
# print(data['parking_options'].isnull().sum()) --> same as laundry_options column

# preprocessing labels into integer typr which are of string type

le = preprocessing.LabelEncoder()
data['type'] = le.fit_transform(data.type)
data['region'] = le.fit_transform(data.region)
data['laundry_options'] = le.fit_transform(data.laundry_options)
data['parking_options'] = le.fit_transform(data.parking_options)


# features used for prediction
features = ['type', 'region', 'sqfeet', 'beds', 'baths', 'comes_furnished',
            'laundry_options', 'parking_options']
X = data[features]  # features/ columns used for prediction

# splitting data into training data and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# creating our model using RandomForestRegressor()
rent_model = DecisionTreeRegressor(random_state=1)  # defining our model
rent_model.fit(train_X, train_y)  # making model fit for prediction

preds = rent_model.predict(val_X)  # predicting the values

# comparing predicted values and original values
print("Predictions are: ", rent_model.predict(val_X.head()))
print("Original values are: ", val_y.head().tolist())
print("Mean Absolute Error is: ", mean_absolute_error(val_y, preds))
