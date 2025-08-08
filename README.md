✅ Minimal README.md for NumPy Beginners
# NumPy for Absolute Beginners 🧮

NumPy (Numerical Python) is a powerful Python library for numerical computations. This repository is created for absolute beginners to understand and practice NumPy step by step.

---

## 🔹 What You'll Learn

- What is NumPy and why use it?
- How to install NumPy
- Creating 1D, 2D, and 3D arrays
- Array indexing and slicing
- Mathematical operations on arrays
- Array reshaping and dimensions
- Generating random numbers
- Finding mean, median, standard deviation, etc.
- Broadcasting in NumPy

---

## 🔹 Installation

Install NumPy using pip:

```bash
pip install numpy
Or using conda:

bash
Copy
Edit
conda install numpy
🔹 Basic Example
python
Copy
Edit
import numpy as np

# Create a 1D array
a = np.array([1, 2, 3])
print("1D Array:", a)

# Create a 2D array
b = np.array([[1, 2], [3, 4]])
print("2D Array:\n", b)

# Perform element-wise addition
print("Add 10:", a + 10)
🔹 Useful NumPy Functions
python
Copy
Edit
np.zeros((2, 3))        # Create array of all zeros
np.ones((3, 3))         # Create array of all ones
np.eye(3)               # Identity matrix
np.arange(0, 10, 2)     # Array from 0 to 10 with step 2
np.linspace(0, 1, 5)    # 5 numbers between 0 and 1
np.random.rand(2, 2)    # Random 2x2 array
np.mean(a)              # Mean of array
np.std(a)               # Standard deviation
🔹 Why Use NumPy?
Faster than regular Python lists

Supports multi-dimensional arrays

Easy to use for mathematical and statistical operations

Foundation for libraries like Pandas, Scikit-learn, TensorFlow, etc.

🔹 Next Steps
After learning NumPy, explore:
Pandas for data analysis
Matplotlib/Seaborn for data visualization
Scikit-learn for machine learning

----------------------------------------------------------------------------------------------------------------------

# 📘 pandas Quick Revision Notes 🧠

import pandas as pd  # Always import like this

# 🧱 Core Data Structures
pd.Series([1, 2, 3])  # 1D array (Series)
pd.DataFrame({"A": [1, 2], "B": [3, 4]})  # 2D table (DataFrame)

# 📥 Read Data
pd.read_csv("file.csv")
pd.read_excel("file.xlsx")
pd.read_json("file.json")

# 📤 Write Data
df.to_csv("out.csv", index=False)
df.to_excel("out.xlsx", index=False)

# 🔍 Data Info
df.head()           # Top 5 rows
df.tail()           # Last 5 rows
df.shape            # Rows, Columns
df.info()           # Types, nulls
df.describe()       # Stats summary
df.columns          # Column names

# 🎯 Select Data
df["Col"]           # Single column
df[["Col1", "Col2"]]# Multiple columns
df.loc[0]           # Row by label
df.iloc[0]          # Row by index
df[df["Age"] > 25]  # Filter rows

# ➕ Add / ❌ Drop / ✏️ Rename
df["NewCol"] = [1,2]
df.drop("Col", axis=1, inplace=True)
df.rename(columns={"Old":"New"}, inplace=True)

# 🚫 Missing Data
df.isnull()         # Check nulls
df.dropna()         # Remove nulls
df.fillna(0)        # Replace nulls

# 🔢 Aggregation
df.sum(), df.mean(), df.count(), df.min(), df.max()

# 🔘 GroupBy
df.groupby("Gender").mean()
df.groupby(["Dept", "Gender"]).sum()

# 🔀 Sort
df.sort_values("Age")
df.sort_values(["Age", "Marks"], ascending=[True, False])

# 🔗 Join / Merge / Concat
pd.concat([df1, df2])
pd.merge(df1, df2, on="ID")
pd.merge(df1, df2, how="left")

# 🧹 Clean & Convert
df["Col"] = df["Col"].astype(int)
df["Name"] = df["Name"].str.lower()
df["Date"] = pd.to_datetime(df["Date"])

# 📆 Date Handling
df["Date"].dt.year
df["Date"].dt.month
df["Date"].dt.day_name()

# 🎯 Misc
df.duplicated()
df.drop_duplicates()
df.nunique()
df.sample(5)

# ✅ Tips
# - Use inplace=True for direct changes
# - Prefer vectorized ops over loops
# - Always check df.info() and df.describe()

