from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 3: Save the model to a file
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as iris_model.pkl")
