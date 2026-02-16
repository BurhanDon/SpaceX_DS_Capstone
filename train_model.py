# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load Data
# Assuming dataset_part_2.csv is in the same directory
data = pd.read_csv("dataset_part_2.csv")

# 2. Define Features and Target
# We select features that are easy for a user to input in an app
features = ['PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'Block', 'ReusedCount']
target = 'Class'

X = data[features]
y = data[target]

# 3. Create Preprocessing Pipeline
# Numeric features need scaling
numeric_features = ['PayloadMass', 'Flights', 'Block', 'ReusedCount']
numeric_transformer = StandardScaler()

# Categorical features need One-Hot Encoding
categorical_features = ['Orbit', 'LaunchSite']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Boolean features (GridFins, Reused, Legs) are already 0/1 or True/False, we pass them through
# but let's ensure they are treated correctly.
pass_through_features = ['GridFins', 'Reused', 'Legs']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('passthrough', 'passthrough', pass_through_features)
    ])

# 4. Create the Full Pipeline (Preprocessor + Classifier)
# We use Logistic Regression as it's robust and gives probabilities
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=0.01, solver='lbfgs'))
])

# 5. Train the Model
print("Training model...")
pipeline.fit(X, y)
print("Model trained.")

# 6. Save the Model
joblib.dump(pipeline, 'spacex_model.pkl')
print("Model saved as 'spacex_model.pkl'")