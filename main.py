import pandas as pd
from sklearn.model_selection import train_test_split

from src.knn import use_knn

weatherdata = pd.read_csv('./data/seattle-weather.csv')

weatherdata = weatherdata[weatherdata['weather'] != "fog"]

# Merge drizzle => rain
weatherdata['weather'] = weatherdata['weather'].replace({
    'drizzle': 'rain',
    'snow': 'rain'
})

print(weatherdata.head())

feature_columns = [
    "temp_max",
    "temp_min",
    "precipitation",
    "wind",
]
X = weatherdata[feature_columns]
y = weatherdata['weather']

print("All column names:")
print(weatherdata.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

use_knn(X_train, X_test, y_train, y_test)

