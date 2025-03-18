pip install numpy

import numpy as np
arr=np.array([1,2,3,4,5])
print(arr)
print(type(arr))

import numpy as np
arr=np.array([1,2,3,4,5])
print(arr)
print(type(arr))

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)

import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[0])

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[2],arr[3])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr)
print('2nd element on 1st row: ', arr[0, 1])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('5th element on 2nd row: ', arr[1, 4])

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[1, 1, 1])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:3])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-2])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:6:2])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:5:2])

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0, 1:4])

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1,0:2])

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2,3])

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 1:4])

import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape)

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(3,4)

print(newarr)

# add matrices
from numpy import array
A = array([[1, 2, 3], [4, 5, 6]])
print(A)
B = array([[1, 2, 3], [4, 5, 6]])
print(B)
C = A + B
print(C)

# element-wise multiply matrices
from numpy import array
A = array([[1, 2, 3], [4, 5, 6]])
print(A)
B = array([[1, 2, 3], [4, 5, 6]])
print(B)
C = A * B
print(C)

from numpy import random

x = random.randint(100)

print(x)

from numpy import random

x = random.rand()

print(x)


from numpy import random

x=random.randint(100, size=(5))

print(x)


from numpy import random

x=random.randint(100, size=(5))

print(x)

from numpy import random

x = random.randint(100, size=(3, 5))

print(x)


EXPERIMENTx2 BY UDAY

/content/cocoa.csv

import pandas as pd
import numpy as np

# Creating empty series
ser = pd.Series()
#print("Pandas Series: ", ser)

# simple array
data = np.array(['g', 'p', 'r', 'e', 'c'])

ser = pd.Series(data)
print("Pandas Series:\n", ser)


import numpy as np
import pandas as pd

info= np.array(['p','a','n','d','a','s'])
#print(info)
ser=pd.Series(info)
print(ser)

# import pandas and numpy
import pandas as pd
import numpy as np

# creating simple array
data = np.array(['g','p','r','e','c','k', 'u','r','n','o','o','l'])
ser = pd.Series(data)
#print(ser)


#retrieve the first element
print(ser[:5])

# import pandas as pd
import pandas as pd

df = pd.DataFrame()

print(df)

# import pandas as pd
import pandas as pd

# list of strings
lst = ['Assam', 'Andhra Pradesh', 'Bhopal', 'Delhi',
            'Maharastra', 'Tamilnadu', 'Karnataka']

df = pd.DataFrame(lst)
print(df)

# DataFrame from dictonary  / lists


import pandas as pd

# initialise data of lists.
data = {'Name':['Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 19, 18]}

# Create DataFrame
df = pd.DataFrame(data)

# Print the output.
print(df)


data['Age']

# importing pandas as pd
import pandas as pd

# dictionary of lists
dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score':[90, 40, 80, 98]}

df = pd.DataFrame(dict)

print(df)


import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)

import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df)

import pandas as pd
data=pd.read_csv("/content/cocoa.csv")
data

data.tail()
data.head()

# importing pandas as pd
import pandas as pd

# importing numpy as np
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score':[52, 40, 80, 98],
        'Fourth Score':[np.nan, np.nan, np.nan, 65]}

# creating a dataframe from dictionary
df = pd.DataFrame(dict)
#df

# using dropna() function
df.dropna()

# importing pandas as pd
import pandas as pd

# importing numpy as np
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

# creating a dataframe from dictionary
df = pd.DataFrame(dict)

# filling missing value using fillna()
df.fillna(0)

# importing pandas as pd
import pandas as pd

# importing numpy as np
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

# creating a dataframe from list
df = pd.DataFrame(dict)
#print(df)

# using isnull() function
df.isnull()

# Import pandas package
import pandas as pd

# Define a dictionary containing employee data
data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Age':[27, 24, 22, 32],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd']}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)
#print(df)

# select two columns
print(df[['Name', 'Qualification',"Age"]])

import pandas as pd
data = { "calories": [420, 380, 390], "duration": [50, 40, 45]};
myvar = pd.DataFrame(data)
print(myvar)

import pandas as pd
data = { "calories": [420, 380, 390], "duration": [50, 40, 45]};
myvar = pd.Series(data)
print(myvar)

import pandas as pd
calories = {"day1": 420, "day2": 380, "day3": 390};
myvar = pd.Series(calories, index = ["day1", "day2"]);
print(myvar)

df.info()

df.describe()

df.isnull().sum()

