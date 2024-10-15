import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Change this line
import pickle

# Generating synthetic data for demonstration
data = {
    'gre_score': np.random.randint(290, 340, 100),
    'toefl_score': np.random.randint(90, 120, 100),
    'university_rating': np.random.randint(1, 6, 100),
    'sop': np.random.rand(100) * 5,
    'lor': np.random.rand(100) * 5,
    'cgpa': np.random.rand(100) * 10,
    'research': np.random.randint(0, 2, 100),  # 0 or 1 for no or yes
    'admission_chance': np.random.rand(100)  # Target variable
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research']]
y = df['admission_chance']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()  # Change this line
model.fit(X_train, y_train)

# Saving the model to a pickle file
filename = 'finalized_model.pickle'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model saved as {filename}')
