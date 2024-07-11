from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
df=pd.read_csv(r'cereal.csv')
df.info()
print("\nNull values in each column:\n", df.isnull().sum())
if df.isnull().sum().sum() > 0:
  # Display heatmap of missing values before imputation
  plt.figure(figsize=(10, 8))
  sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
  plt.title('Missing Values Heatmap (Before)')
  plt.show()

  # Impute missing values with the mean
  imputer = SimpleImputer(strategy='mean')
  df = imputer.fit_transform(df)  # Create a transformed DataFrame

  print("\nNull values after imputation:\n", df.isnull().sum())

  # Display heatmap of missing values after imputation
  plt.figure(figsize=(10, 8))
  sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
  plt.title('Missing Values Heatmap (After)')
  plt.show()
# Remove non-numeric columns for correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

# Display a heatmap to visualize the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()

# Box plot (example: sugars by cereal type)
plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='sugars', data=df)
plt.xlabel('Cereal Type')
plt.ylabel('Sugars (grams)')
plt.title('Distribution of Sugars by Cereal Type (Box Plot)')
plt.show()

# Pairplot (visualizes relationships between all numerical features)
sns.pairplot(df.select_dtypes(include=[np.number]))
plt.show()

# Convert 'type' to binary
df['type'] = (df['type'] == 'C').astype(int)
# Display unique values of 'mfr'
print("\nUnique values in 'mfr':\n", df['mfr'].unique())
# Replace -1 with NaN and fill with mean values for specific columns
df = df.replace(-1, np.NaN)
for col in ['carbo', 'sugars', 'potass']:
    df[col] = df[col].fillna(df[col].mean())
df.drop('name',axis=1,inplace=True)
# Convert 'mfr' to dummy variables
dummy = pd.get_dummies(df['mfr'], dtype=int)
df = pd.concat([df, dummy], axis=1)
df.drop('mfr', axis=1, inplace=True)
# Separate features and target variable
y = df['rating']
X = df.drop('rating', axis=1)
# Standardize the features
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train linear regression models
lr = LinearRegression()
r = Ridge(alpha=1.5)
l = Lasso(alpha=0.001)
lr.fit(X_train, y_train)
r.fit(X_train, y_train)
l.fit(X_train, y_train)
# Train decision tree regressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Train random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the models as pickle files
with open('linear_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
    
with open('ridge_regression.pkl', 'wb') as f:
    pickle.dump(r, f)
    
with open('lasso_regression.pkl', 'wb') as f:
    pickle.dump(l, f)
    
with open('decision_tree_regressor.pkl', 'wb') as f:
    pickle.dump(dt, f)
    
with open('random_forest_regressor.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Evaluate models
print(f"Linear Regression score: {lr.score(X_test, y_test):.4f}")
print(f"Ridge Regression score: {r.score(X_test, y_test):.4f}")
print(f"Lasso Regression score: {l.score(X_test, y_test):.4f}")
print(f"Decision Tree Regressor score: {dt.score(X_test, y_test):.4f}")
print(f"Random Forest Regressor score: {rf.score(X_test, y_test):.4f}")

# MODEL EVALUATION
# Make predictions on test set
y_pred_lr = lr.predict(X_test)
y_pred_r = r.predict(X_test)
y_pred_l = l.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Calculate evaluation metrics for each model
models = ["Linear Regression", "Ridge Regression", "Lasso Regression",
          "Decision Tree Regressor", "Random Forest Regressor"]
y_preds = [y_pred_lr, y_pred_r, y_pred_l, y_pred_dt, y_pred_rf]
for model, y_pred in zip(models, y_preds):
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Square root for interpretability
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Percentage error

    print(f"\nModel: {model}")
    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")