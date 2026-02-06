# Sales Prediction Using Machine Learning in Python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("advertising.csv")
print("Dataset Preview:")
print(data.head())

# -----------------------------
# 2. Feature and Target Split
# -----------------------------
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Model Training
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("------------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# 7. Example Prediction
# -----------------------------
example = pd.DataFrame({
    "TV": [150],
    "Radio": [25],
    "Newspaper": [30]
})

predicted_sales = model.predict(example)
print("\nPredicted Sales for given advertising budget:", round(predicted_sales[0], 2))
