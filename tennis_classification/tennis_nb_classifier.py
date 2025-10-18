import pandas as pd
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Step 1: Read the Excel file
excel_file = 'tennis_dataset.xlsx'  # Change this to your file name

try:
    df = pd.read_excel(excel_file, sheet_name=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
except FileNotFoundError:
    print(f"ERROR: File '{excel_file}' not found!")
    exit()
except Exception as e:
    print(f"ERROR reading Excel file: {e}")
    exit()

# Step 2: Encode categorical values to numbers
# First, convert all string columns to lowercase
df['outlook'] = df['outlook'].str.lower()
df['temp'] = df['temp'].str.lower()
df['humidity'] = df['humidity'].str.lower()
df['wind'] = df['wind'].str.lower()
df['play_tennis'] = df['play_tennis'].str.lower()

# Create encoding dictionaries for each feature
outlook_encoding = {'sunny': 0, 'overcast': 1, 'rain': 2}
temp_encoding = {'hot': 2, 'mild': 1, 'cold': 0}
humidity_encoding = {'high': 1, 'normal': 0}
wind_encoding = {'weak': 0, 'strong': 1}
play_encoding = {'yes': 1, 'no': 0}

# Apply encoding
df['outlook_encoded'] = df['outlook'].map(outlook_encoding)
df['temp_encoded'] = df['temp'].map(temp_encoding)
df['humidity_encoded'] = df['humidity'].map(humidity_encoding)
df['wind_encoded'] = df['wind'].map(wind_encoding)
df['play_tennis_encoded'] = df['play_tennis'].map(play_encoding)

# Check for any NaN values after encoding
print("\nChecking for encoding issues:")
print(f"outlook_encoded NaN count: {df['outlook_encoded'].isna().sum()}")
print(f"temp_encoded NaN count: {df['temp_encoded'].isna().sum()}")
print(f"humidity_encoded NaN count: {df['humidity_encoded'].isna().sum()}")
print(f"wind_encoded NaN count: {df['wind_encoded'].isna().sum()}")
print(f"play_tennis_encoded NaN count: {df['play_tennis_encoded'].isna().sum()}")

if df[['outlook_encoded', 'temp_encoded', 'humidity_encoded', 'wind_encoded', 'play_tennis_encoded']].isna().any().any():
    print("\n WARNING: Some values couldn't be encoded. Check your data:")
    print(df[df.isna().any(axis=1)])
    exit()

print("✓ All values encoded successfully!")
print("="*60)

# Step 3: Separate features (X) and target (y)
X = df[['outlook_encoded', 'temp_encoded', 'humidity_encoded', 'wind_encoded']]
y = df['play_tennis_encoded']

# Step 4: Train the Naive Bayes classifier
model = CategoricalNB()
model.fit(X, y)

# Step 5: Create the test instance for Day 15
# Given: outlook=sunny, temperature=cold, humidity=high, wind=strong
test_instance = pd.DataFrame({
    'outlook_encoded': [0],     # sunny
    'temp_encoded': [0],        # cold
    'humidity_encoded': [1],    # high
    'wind_encoded': [1]         # strong
})

# Step 6: Make prediction
prediction = model.predict(test_instance)
probabilities = model.predict_proba(test_instance)

# Step 7: Display results
print("="*60)
print("NAIVE BAYES TENNIS PREDICTION")
print("="*60)
print("\nTraining Data Summary:")
print(f"Total days: {len(df)}")
print(f"Days played tennis: {sum(y == 1)}")
print(f"Days didn't play tennis: {sum(y == 0)}")

print("\n" + "="*60)
print("TEST INSTANCE (DAY 15):")
print("="*60)
print("Outlook:     SUNNY")
print("Temperature: COLD")
print("Humidity:    HIGH")
print("Wind:        STRONG")

print("\n" + "="*60)
print("PREDICTION RESULTS:")
print("="*60)
play_decision = "YES, WILL PLAY TENNIS" if prediction[0] == 1 else "NO, WILL NOT PLAY TENNIS"
print(f"\nPrediction: {play_decision}")
print(f"\nProbability of NOT playing: {probabilities[0][0]:.4f} ({probabilities[0][0]*100:.2f}%)")
print(f"Probability of PLAYING:     {probabilities[0][1]:.4f} ({probabilities[0][1]*100:.2f}%)")

# Step 8: Show similar days from training data
print("\n" + "="*60)
print("SIMILAR DAYS FROM TRAINING DATA:")
print("="*60)
similar = df[(df['outlook'] == 'sunny') & 
             (df['temp'] == 'cold') & 
             (df['humidity'] == 'high') & 
             (df['wind'] == 'strong')]
if not similar.empty:
    print("Exact matches found:")
    print(similar[['day', 'outlook', 'temp', 'humidity', 'wind', 'play_tennis']])
else:
    print("No exact matches found. Showing days with sunny outlook and high humidity:")
    similar_partial = df[(df['outlook'] == 'sunny') & (df['humidity'] == 'high')]
    if not similar_partial.empty:
        print(similar_partial[['day', 'outlook', 'temp', 'humidity', 'wind', 'play_tennis']])

# Step 9: Model insights
print("\n" + "="*60)
print("MODEL INSIGHTS:")
print("="*60)
print("\nPrior Probabilities:")
print(f"P(Play Tennis) = {np.exp(model.class_log_prior_[1]):.4f}")
print(f"P(Don't Play) = {np.exp(model.class_log_prior_[0]):.4f}")

print("\nFeature Analysis:")
print("Based on training data patterns:")
print(f"- Sunny days: {len(df[df['outlook']=='sunny'])} occurrences")
print(f"- High humidity: {len(df[df['humidity']=='high'])} occurrences")
print(f"- Strong wind: {len(df[df['wind']=='strong'])} occurrences")
print(f"- Cold temperature: {len(df[df['temp']=='cold'])} occurrences")