import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataSet_ex4.csv")
print("the first 5 values of data \n", data.head())

x = data.iloc[:, :-1]
"""this is without the last column which is target (Yes or no)"""
print("the first 5 values of train data\n", x.head())


y = data.iloc[:, -1]
print("\nthe first 5 values of train data are: \n",y.head())


"""It maps each unique category (label) in a column to a unique integer
['Sunny', 'Rain', 'Overcast', 'Sunny', 'Rain'] → [2, 1, 0, 2, 1]
"""
le_outlook = LabelEncoder()
x['Outlook'] = le_outlook.fit_transform(x['Outlook'])

le_temp = LabelEncoder()
x['Temperature'] = le_temp.fit_transform(x['Temperature'])

le_humidity = LabelEncoder()
x['Humidity'] = le_humidity.fit_transform(x['Humidity'])

le_windy = LabelEncoder()
x['Wind'] = le_windy.fit_transform(x['Wind'])


print("\nNow the Train data is:\n", x.head())

le_play = LabelEncoder()
y = le_play.fit_transform(y)

print("\nNow the Train output is:\n", y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20
)


classifier = GaussianNB()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy is:", accuracy)