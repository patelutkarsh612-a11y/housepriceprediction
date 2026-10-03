import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Input features
X = data[["area", "bedrooms", "bathrooms", "parking", "age"]]

# Target
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("R2 Score:", accuracy)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("model.pkl created successfully!")