import pandas as pd

print("=== Fake Reviews Dataset ===")
df1 = pd.read_csv(r'd:\TrustLens\Dataset\Fake and Real Product Reviews\fake reviews dataset.csv', nrows=2)
print(df1.columns.tolist())
print(df1['label'].dtype, "| unique:", df1['label'].unique())

print("\n=== Amazon Fine Food Reviews ===")
df2 = pd.read_csv(r'd:\TrustLens\Dataset\Amazon Fine Food Reviews\Reviews.csv', nrows=2)
print(df2.columns.tolist())

print("\n=== Amazon Products Dataset ===")
df3 = pd.read_csv(r'd:\TrustLens\Dataset\Amazon Products Dataset 2023\amazon_products.csv', nrows=2)
print(df3.columns.tolist())
print(df3.iloc[0].to_string())
