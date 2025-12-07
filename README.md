# EDA-Analysis
Working on EDA Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data
df = pd.read_csv(r'C:\Users\USER\Documents\GSS.csv', encoding='latin1')
df.columns = df.columns.str.strip()  # remove extra spaces
print("Columns:", df.columns)
print("Initial Data:")
print(df.head())

# Step 2: Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

for col in ['Cost', 'Sales']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

# Step 3: Data Transformation
if 'Sales' in df.columns and 'Cost' in df.columns:
    df['Profit'] = df['Sales'] - df['Cost']

if 'Order Date' in df.columns:
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year

# Step 4: Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Step 5: Visualization
# 5a: Monthly Profit
if 'Month' in df.columns and 'Profit' in df.columns:
    monthly_profit = df.groupby('Month')['Profit'].sum().reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x='Month', y='Profit', data=monthly_profit, palette='viridis')
    plt.title('Monthly Profit')
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.show()

# 5b: Profit by Category
if 'Category' in df.columns and 'Profit' in df.columns:
    df['Category'] = df['Category'].astype(str).fillna('Unknown')
    plt.figure(figsize=(8,5))
    sns.barplot(x='Category', y='Profit', data=df, estimator=sum, ci=None, palette='magma')
    plt.title('Profit by Category')
    plt.show()

# 5c: Sales vs Profit scatter plot
plt.figure(figsize=(8,5))
if 'Category' in df.columns:
    sns.scatterplot(x='Sales', y='Profit', data=df, hue='Category', palette='Set2')
else:
    sns.scatterplot(x='Sales', y='Profit', data=df)
plt.title('Sales vs Profit')
plt.show()

# 5d: Correlation heatmap
if all(col in df.columns for col in ['Sales','Cost','Profit']):
    plt.figure(figsize=(8,6))
    sns.heatmap(df[['Sales','Cost','Profit']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
