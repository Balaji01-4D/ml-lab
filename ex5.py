import pandas as pd

msg = pd.read_csv("document.csv", names=["message", "label"])
print("The total instance of message")
print(msg.shape[0])
msg["labelnum"] = msg.label.map({"pos": 1, "neg": 0})

x = msg.messge
y = msg.labelnum

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.feature_extraction.text import CountVectorizer

count_v = CountVectorizer()
xTrain_dm = count_v.fit_transform(x_train)
xTest_dm = count_v.fit_transform(x_test)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(xTrain_dm, y_train)

pred = model.predict(xTest_dm)


from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score
print("Accuracy scores: ")
print("Recall score", recall_score(y_train, pred))