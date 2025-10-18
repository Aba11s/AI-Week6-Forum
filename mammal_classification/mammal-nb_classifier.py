import pandas as pd
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Step 1: Read the Excel file
excel_file = 'animal_dataset.xlsx'

try:
    df = pd.read_excel(excel_file, sheet_name=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # remove any empty col
    
    print(f"✓ Successfully loaded {excel_file}")
    print(f"✓ Dataset has {len(df)} rows and {len(df.columns)} columns")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print("\n" + "="*60)
    
except FileNotFoundError:
    print(f"ERROR: File '{excel_file}' not found!")
    exit()
except Exception as e:
    print(f"ERROR reading Excel file: {e}")
    exit()

# Step 2: Encode categorical values to numbers
# yes=2, sometimes=1, no=0 for features
# mammals=1, non-mammals=0 for target
encoding = {'yes': 2, 'sometimes': 1, 'no': 0}

df['give_birth_encoded'] = df['give_birth'].map(encoding)
df['can_fly_encoded'] = df['can_fly'].map(encoding)
df['live_in_water_encoded'] = df['live_in_water'].map(encoding)
df['have_legs_encoded'] = df['have_legs'].map(encoding)
df['class_encoded'] = df['class'].map({'mammals': 1, 'non-mammals': 0})

# Step 3: Separate features (X) and target (y)
X = df[['give_birth_encoded', 'can_fly_encoded', 'live_in_water_encoded', 'have_legs_encoded']]
y = df['class_encoded']

# Step 4: Train the Naive Bayes classifier
model = CategoricalNB()
model.fit(X, y)

# Step 5: Create the test instance
test_instance = pd.DataFrame({
    'give_birth_encoded': [2],  # yes
    'can_fly_encoded': [0],     # no
    'live_in_water_encoded': [2],  # yes
    'have_legs_encoded': [0]    # no
})

# Step 6: Make prediction
prediction = model.predict(test_instance)
probabilities = model.predict_proba(test_instance)

# Step 7: Display results
print("="*60)
print("NAIVE BAYES ANIMAL CLASSIFICATION")
print("="*60)
print("\nTraining Data Summary:")
print(f"Total animals: {len(df)}")
print(f"Mammals: {sum(y == 1)}")
print(f"Non-mammals: {sum(y == 0)}")

print("\n" + "="*60)
print("PREDICTION RESULTS:")
print("="*60)
class_name = "MAMMAL" if prediction[0] == 1 else "NON-MAMMAL"
print(f"\nPredicted Class: {class_name}")
print(f"\nProbability of being Non-Mammal: {probabilities[0][0]:.4f} ({probabilities[0][0]*100:.2f}%)")
print(f"Probability of being Mammal:     {probabilities[0][1]:.4f} ({probabilities[0][1]*100:.2f}%)")

# Step 8: Show some examples from training data that match
print("\n" + "="*60)
print("SIMILAR ANIMALS FROM TRAINING DATA:")
print("="*60)
similar = df[(df['give_birth'] == 'yes') & 
             (df['can_fly'] == 'no') & 
             (df['live_in_water'].isin(['yes', 'sometimes'])) & 
             (df['have_legs'] == 'no')]
if not similar.empty:
    print(similar[['name', 'give_birth', 'can_fly', 'live_in_water', 'have_legs', 'class']])
else:
    print("No exact matches found in training data")

# Step 9: Model insights
print("\n" + "="*60)
print("MODEL INSIGHTS:")
print("="*60)
print("\nPrior Probabilities:")
print(f"P(Mammal) = {np.exp(model.class_log_prior_[1]):.4f}")
print(f"P(Non-Mammal) = {np.exp(model.class_log_prior_[0]):.4f}")