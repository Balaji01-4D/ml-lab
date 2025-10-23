import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()

x = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)

standardscalar = StandardScaler()
scaled_x = standardscalar.fit_transform(x)

pca = PCA(n_components=2)
pca_x = pca.fit_transform(scaled_x)

for target, color in zip([0, 1, 2], ["red", "green", "blue"]):
    plt.scatter(
        pca_x[y["Target"] == target, 0],
        pca_x[y["Target"] == target, 1],
        c=color,
        label=data.target_names[target],
    )


x_train, x_test, y_train, y_test = train_test_split(
    pca_x, y["Target"], random_state=42, test_size=0.3
)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, y_train)

y_pred = kn.predict(x_test)

score = accuracy_score(y_test, y_pred)
print(score)
