import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

#LOAD THE DATASET
data = pd.read_csv("C:\\Users\\tonga\\PycharmProjects\\DataMining-Project\\data\\Apartment Prices.csv")

#SPLIT INTO CATEGORICAL, NUMERIC AND TARGET VARIABLES
categorical = data.select_dtypes(include='object')
numeric = data.select_dtypes(include=np.number).drop('PRICE (GEL)', axis=1)
target = data['PRICE (GEL)']

#HANDLE MISSING VALUES
for i in numeric.columns:
  numeric[i] = numeric[i].fillna(numeric[i].median())

categorical = pd.get_dummies(categorical, dummy_na=True, dtype=int)

new_data = pd.concat([numeric, categorical, target], axis=1)
print(new_data)
print(new_data.isnull().sum())

new_data.to_csv("C:\\Users\\tonga\\PycharmProjects\\DataMining-Project\\data\\apartment_prices.csv", index=False)