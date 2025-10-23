import numpy as np
import pandas as pd
from pgmpy.model import BayesianNetwork
from pgmpy.estimator import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

np.random.seed(42)

data_size = 1000

data = pd.DataFrame({
    "fever": np.random.choice([0, 1], size=data_size, p=[0.3, 0.7]),
    "cough": np.random.choice([0, 1], size=data_size, p=[0.4, 0.6]),
    "fatigue": np.random.choice([0, 1], size=data_size, p=[0.5, 0.5]),
    "travel_history": np.random.choice([0, 1], size=data_size, p=[0.8, 0.2]),
})


data["covid"] = np.where(
    (data["fever"] & data["cough"] & (data["fatigue"] | data["travel_history"])),
    1,
    0
)

# data.to_csv("data.csv", index=False)

model = BayesianNetwork([
    ("fever", "covid"),
    ("cough", "covid"),
    ("fatigue", "covid"),
    ("travel_history", "covid")
])

model.fit(data, estimator=MaximumLikelihoodEstimator)

print("Model stucuture")
print(model.edges())

inference = VariableElimination(model)

query = inference.query(variable=["covid"], evidence={
    "fever":1, "cough":1, "travel_history":1
})
print("\nCOVID-19 Diagnosis Probability:")
print(query)