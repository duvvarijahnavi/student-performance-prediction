import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())