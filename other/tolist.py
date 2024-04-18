# importing pandas module
import pandas as pd

# importing regex module
import re

# making data frame
data = pd.read_csv("nba.csv")

# removing null values to avoid errors
data.dropna(inplace = True)

# storing dtype before operation
dtype_before = type(data["Salary"])

# converting to list
salary_list = data["Salary"].tolist()

# storing dtype after operation
dtype_after = type(salary_list)

# printing dtype
print("Data type before converting = {}\nData type after converting = {}".format(dtype_before, dtype_after))

# displaying list
print(salary_list)
