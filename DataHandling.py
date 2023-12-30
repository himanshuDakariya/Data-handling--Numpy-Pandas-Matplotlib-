import numpy as np 

# 4.1 The NumPy ndarray: A Multidimensional Array Object



# data=np.random.randn(2,3)
# print(data)

# print(data*10)

# print(data+data)

# print(data.shape)


# ---------------------------------------

# Creating ndarray

# data=[1,2,3,4,5,6,7,8,9,10]

# arr1=np.array(data)
# print(arr1)


# data=[1,2,3,4,5.5,6,7,8,9,10]

# arr1=np.array(data)
# print(arr1)

# data=[[1,2,3],[4,5,6]]

# arr1=np.array(data)
# print(arr1)

# print(arr1.ndim)


# print(np.zeros(10))
# print(np.zeros((10,10)))
# print(np.empty((3,2,2)))

# print(np.arange(10))
#
#
#Type casting np ndarrays

# arr=np.array([1,2,3,4,5])
# print(arr.dtype)

# floatarr=arr.astype(np.float64)
# print(floatarr.dtype)


# arr=np.array([1,9,2.9,3.1])

# intarr=arr.astype(np.int32)
# print(intarr.dtype)



# # -----------------------------------------------------
# #     Copying Case>>>>>>

# arr= np.array([1,2,3,4,5])

# arrchotu=arr[2:]
# print(arrchotu)

# arrchotu[1] = 16
# print(arr)


# If you want to ACTUALLY COPY
# Explicitly mention copy function . . 

# arr= np.array([1,2,3,4,5])

# arrchotu=arr[2:].copy()
# print(arrchotu)

# arrchotu[1] = 16
# print(arr)



# # ---------------------------------------------------------

# arr3d = np.array(
#     [[[1,2,3],[4,5,6]],
#      [[10,20,30],[40,50,60]]]
#  )
# print(arr3d)
# print(arr3d[0])
# print(arr3d[0][1])
# print(arr3d[0][1][0])


# -------------------------------------------------------


# arr3d = np.array(
#     [[[1,2,3],[4,5,6]],
#      [[10,20,30],[40,50,60]]]
#  )
# print(arr3d)
# print(arr3d[0])
# print(arr3d[0][1])
# print(arr3d[0][1][0])

# print(arr3d[0,1,1])



# Both scalar and array can be assigned  


# old= arr3d[0].copy()
# print(old)

# arr3d[0] = 0
# print(arr3d)
# arr3d[0]=old
# print(arr3d)



# arr=np.array([[1,2,3],[4,5,6]])
# print(arr)

# # print(arr[0])
# # print(arr[:2])
# # print(arr[:2,1:])


# print(arr[:,:1])



# -----------------------------------------------


# # Boolean Indexing 

# data=np.array([1,2,3,4])
# names =np.array(['a','a','b','a'])
# print(data)
# print(names == 'a')

# print(data[names == 'a'])



# ----------------------------------------------------

# data= np.array([1,2,34,4,5,6,7])

# print(data[data<3])



# ---------------------------------------

# data=np.array([1,2,3,4])
# names =np.array(['a','a','b','a'])

# print(data[names!='b'])

# ---------------------------------------------


# arr = np.arange(16).reshape(4,4)


# print(arr)
# print(arr[[0,2]])
# print(arr[[-1,-2,-3]])


# arr= np.arange(64).reshape(8,8)
# print(arr)

# print(arr[[1,2,3],[4,5,6]])  # 1st index row ka 4th index elem 
#                              # 2nd index row ka 5th index elem

# print( arr[[1,2,3]][:,[4,5,6]])  #1st index row ke 4,5,6
#                                  #2nd index row ke 4,5,6 index elem...



# ----------------------------------------------

# arr= np.arange(4).reshape(2,2)
# print(np.dot(arr,arr.T))

# -----------------------------------------------



# arr= np.arange(12).reshape(3,2,2)
# print(arr)

# print(np.swapaxes(arr,0,1))

# -----------------------------------------------




# 4.2 Universal Functions: Fast Element-Wise Array Functions

# arr = np.arange(10)

# print(arr)
# # unary ufunc

# print(np.sqrt(arr))  # if '-ve' then 'nan'
# print(np.exp(arr))

# print('-----------------------------')


# #binary ufunc
# x = np.random.randn(8)
# y = np.random.randn(8)
# print(x)
# print(y)

# print(np.maximum(x,y))

 


# ------------------------------------

# multiple arrays can also be returned using 'mod func'

# arr = np.random.randn(7)*5

# print(arr)

# remainder,wholepart = np.modf(arr)

# print(remainder)
# print(wholepart)

# # decomposes elements into frac and integral part

# ------------------------------------------------



# np.sqrt(arr,arr) will store the sqrt values to array itself


# 000000000000000000000000000000000000000000000000000000000000



# Random Number Generation


#       randint()

# x= np.random.randint(10)
# print(x) 
# # or
# x= np.random.randint(0,10)
# print(x)
# x=np.random.randint(0,10,size=(2,2))
# print(x)

# -----------------------------

#       rand()

# x = np.random.rand();
# print(x)
# x = np.random.rand(5);
# print(x)
# x = np.random.rand(10);
# print(x)
# x = np.random.rand(2,2);
# print(x)
# x = np.random.rand(2,2,2);
# print(x)


# ----------------------------

#      randn()

# x = np.random.randn();
# print(x)
# x = np.random.rand(10);
# print(x)
# x = np.random.rand(2,2);
# print(x)
# x = np.random.rand(3,2,2);
# print(x)


# --------------------------------------------




# CHAPTER 5
# Getting Started with pandas
import pandas as pd


# 5.1 Introduction to pandas Data Structures
# Series

# obj = pd.Series([1,2,3,4,5])
# print(obj)


# obj = pd.Series([1,2.5,3,-5])
# print(obj)

# print(obj.values)
# print(obj.index)
# print()
# print(obj.value_counts)

# obj = pd.Series([1,2,3,4,5])
# print(obj)
# obj[2]=100
# print(obj[2])
# print(obj)
# print(obj[obj>4])

# -------------------------------

# dictdata = {'A':100,'Second':200,'C':300,'Fourth':400}
# s=pd.Series(dictdata)
# print(s)

# ind = ['First','Second','Third','Fourth']

# s=pd.Series(dictdata,index= ind)
# print(s)

# ---------------------------------


# s=pd.Series({'A':100,'B':200,'C':300,'D':400})
# print(s)

# s.index.name = 'Alphabet'
# s.name = 'Alpha-Numeric'

# print(s)
# s.index= ['a','b','c','d']
# print(s)

# -------------------------------------


#           DataFrame 


# data= {'States':['Odisha','MP','Bihar','Punjab'],
#        'Population':[5462,8469,7456,1254],
#        'AirQuality':[124,110,152,170]}

# df= pd.DataFrame(data)
# print(df)

# dfm= pd.DataFrame(data,columns=['Population','AirQuality','States'])
# print(dfm)


# df=pd.DataFrame(data,index=['i','ii','iii','iv'])
# print(df)

# -------------------------------------------

# print(df['States'])
# print(df['AirQuality'])


# -------------------------------------------

# print(df.loc['i'])

# -------------------------------------------
# df['Population']= 0
# print(df)
# df['HappinessInd']=-1
# print(df)




# --------------------------------------------


# val = pd.Series([-1.2, -1.5],
#                  index=['i', 'iv'])

# df['Population']= val
# print(df)


# ----------------------------------------------

# df['ABX']=0
# print(df)
# del df['ABX']
# print(df)

# -----------------------------------------------

#     NEsted Dictionary 



# Nestd= {'Col1':{'a':1,'b':2,'c':3},
#         'Col2':{'x':10,'y':20,'z':30},
#         'Col3':{'p':100,'q':200,'r':300}}

# df=pd.DataFrame(Nestd)
# print(df)



# Nestd= {'Col1':{'a':1,'b':2,'c':3},
#         'Col2':{'a':10,'b':20,'c':30},
#         'Col3':{'a':100,'b':200,'c':300}}


# df=pd.DataFrame(Nestd)
# print(df)
# # df=pd.DataFrame(Nestd,index= ['b','c','d'])
# # print(df)

# # print(df.T)



# # print(df['Col1'][:2])   #Col1 ke 0 or 1th element


# df.index.name='Index';
# df.columns.name= 'Columns'

# print(df)
# print(df.values)


# -------------------------------------


# s=pd.Series([1,2,3,4],index= ['a','b','c','d'])
# print(s)
# index=s.index

# # Index objects are : immmutable 
                    
# print(index)


# ------------------------------------------------




# 5.2 Essential Functionality

# data= {'States':['Odisha','MP','Bihar','Punjab'],
#        'Population':[5462,8469,7456,1254],
#        'AirQuality':[124,110,152,170]}

# df=pd.DataFrame(data,index=['i','ii','iii','iv'])
# print(df)

# reindexing 

# df2=df.reindex(['ii','i','iii','iv'])
# print(df2)

# df2=df.reindex(['i','ii','iii','iv'])
# print(df2)


# df3=pd.Series(['blue','purple','yellow'], index=[0,2,4])

# print(df3)

# # df3=df3.reindex(index=[0,1,2,3,4,5])
# # print(df3)
# df1=df3.reindex(index=[0,1,2,3,4,5],method='ffill')
# print(df1)



# ------------------------------------------


# frame = pd.DataFrame(np.arange(9).reshape(3,3),
#                      index=['a','b','c'],
#                      columns=['x','y','z'])

# print(frame)

# c = ['x','Col2','Col3']
# frame= frame.reindex(columns=c)

# print(frame )



# ---------------------------------------------



#       Dropping Entries from an Axis


# s=pd.Series([1,2,3,4],index=['i','ii','iii','iv'])
# print(s)

# # newS= s.drop('ii')
# # print(newS)

# newS=s.drop(['i','ii'])
# print(newS)


# --------------------------------
# DataFrame


# df=pd.DataFrame(np.arange(16).reshape(4,4),
#                 index=['A','B','C','D'],
#                 columns=['C1','C2','C3','C4'])


# df=df.drop(['A'])
# print(df)

# df=df.drop(['B','C'])
# print(df)

# df=df.drop('C1',axis=1)
# print(df)
# df=df.drop('C2',axis='columns')
# print(df)

# OR4

# df.drop('A',inplace=True)
# print(df)



# .     Indexing, Selection, and Filtering


# s=pd.Series([1,2,3,4])
# print(s)
# print(s[2])
# print(s[[1,2]])
# sn=pd.Series([1,2,3,4],index=['a','b','c','d'])
# print(sn)


# print(s[1:3])
# print(sn['b':'c'])


#        DAtaframe 


# data = pd.DataFrame(np.arange(16).reshape((4, 4)),
#         index=['one', 'two', 'three', 'four'],
#         columns=['Ohio', 'Colorado', 'Utah', 'New York'])

# print(data)

# print(data['Ohio'])
# print(data[['Ohio','Utah']])
# print(data[:])
# print(data[:2])
# print(data[data['Ohio']>1])



# ----------    loc and iloc   ---------------


#  If you want to retrieve a specific row, 
#  you can use .loc:


# loc: df.loc[startrow:endrow , startcol:endcol]

# print(data.loc['one'])
# print(data.loc['one':'three'])
# print(data.loc['one',['Ohio','Utah']])
# print(data.loc[:'three','Ohio'])  # First 3 row Ohio ki


# iloc: df.iloc[strtrowind:endrowind , strtcolind:endcolind]

# print(data.iloc[2])
# print(data.iloc[1:3])
# print(data.iloc[2,[1,2,3]])
# print(data.iloc[[1,2],[2,3]])
# print(data.iloc[:,:3])



# ---------------------------------------------------
# Integer Indexes

# s=pd.Series([1,2,3,4],index=['a','b','c','d'])

# print(s)
# print(s[-1])
# print(s[:1])


# # if int index 

# s=pd.Series([1,2,3,4])
# print(s.loc[:1])

# print(s.iloc[:1])


# Function Application and Mapping



# df= pd.DataFrame(np.arange(9).reshape(3,3),columns=['a','b','c'],index=['x','y','z'])
# # print(np.abs(df))
# print(df)


# f=lambda x: x.max()-x.min()

# ------------------------------------------------
# print(df.apply(f))        # Col wise

# print(df.apply(f,axis=1)) # Rows wise!!!!
# -------------------------------------------------

# def f(x):
#     return pd.Series([x.min(), x.max()], 
#                      index=['min', 'max'])

# print(df.apply(f))




# ----------------------------------------------


# Sorting and Ranking

# s=pd.Series([1,2,3,4],index=['a','c','b','d'])
# print(s)
# print(s.sort_index())




# df= pd.DataFrame(np.arange(9).reshape(3,3),
#                  columns=['a','b','c'],
#                  index=['y','x','z'])

# print(df.sort_index())
# print(df.sort_index(axis=1))
# print(df.sort_index(axis=1,ascending=False))




# s=pd.Series([1,np.nan,3,4],index=['a','c','b','d'])
# print(s.sort_values())




# ----------------------------------------------

# Ranking
# df=pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
# print(df)
# # print(df.sort_values(by='b'))
# print(df.sort_values(by=['a','b']))

# s= pd.Series([7, -5, 7, 4, 2, 0, 4])

# print(s)
# print(s.rank())  # mean ranks 

# print(s.rank(method='first'))
# print(s.rank(method='max'))


# df= pd.DataFrame({'b': [4.3, 7, -3, 2],
#                       'a': [0, 1, 0, 1],
#                       'c': [-2, 5, 8, -2.5]})

# print(df)
# print(df.rank())




# ----------------------------------------------------

# Axis Indexes with Duplicate Labels4


# s=pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])

# print(s['a'])
# print(s.index.is_unique)

# same with dataframe 



# -------------------------------------------------------------

# 5.3 Summarizing and Computing Descriptive Statistics

# df=pd.DataFrame(np.arange(9).reshape(3,3),
#                 columns=['c1','c2','c3'],
#                 index=['x','y','z'])

# print(df)

# print(df.sum())

# print(df.sum(axis=1)).

# df=pd.DataFrame({'b': [4, np.nan, -3, 2], 'a': [0, 1, 0, np.nan]})
# print(df)

# print(df.sum())

# print(df.idxmin())  #returns index where minimum

# print(df.cumsum())  #summulative sum at end of each col



# -----------------------------------------------------

# Correlation and Covariance


# df1=pd.DataFrame(np.arange(9).reshape(3,3),
#                 columns=['One','Two','Three'],
#                 index=['x','y','z'])

# df2=pd.DataFrame(np.arange(9,18).reshape(3,3),
#                 columns=['One','Two','Three'],
#                 index=['x','y','z'])

# print(df1)
# print(df2)


# print(df1['One'].corr(df2['Two']))
# # OR
# print(df1.One.corr(df2.Two))

# print(df1.corrwith(df2))  # Pairwise check 

# -------------------------------------------------------
# Unique Values, Value Counts, and Membership

# s=pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])


# print(s)
# print(s.unique())
# print(s.value_counts())
# print(pd.value_counts(s.values,sort= False))



# to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
# unique_vals = pd.Series(['c', 'b', 'a'])
# print(pd.Index(unique_vals).get_indexer(to_match))



# DataFrame 

# data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
#                      'Qu2': [2, 3, 1, 2, 3],
#                      'Qu3': [1, 5, 2, 4, 4]})
# print(data)

# result = data.apply(pd.value_counts).fillna(0)
# print(result)


# 0000000000000000000000000000000000000000000000000000000000000000
# CHAPTER 6
# Data Loading, Storage, and File Formats



# df=pd.read_csv('D:\Practice\Book1.csv')
# print(df)
# df=pd.read_csv('D:\Practice\Book1.csv',header=None)
# print(df)
# df=pd.read_csv('D:\Practice\Book1.csv', names=['a','b','c'])
# print(df)

# df=pd.read_csv('D:\Practice\Book1.csv', index_col='Name')
# print(df)


# df=pd.read_csv('D:\Practice\Book1.csv', index_col=['Name','Class'])
# print(df)
# print(df.loc['Himanshu'])

# df=pd.read_csv('D:\Practice\Book1.csv', skiprows=[0,1])
# print(df)


# # na_values

# df=pd.read_csv('D:\Practice\CSv.csv',na_values={'message': ['foo', 'NA'], 'something': ['two']})
# print(df)

# ----------------------------------------------


# df=pd.read_csv('D:\Practice\Book1.csv',nrows=2)
# print(df)

# # chunker is an iterator that you can loop through.
# # In each iteration, chunk will be a DataFrame 
# # containing the next 1000 rows of your data.


# chunker = pd.read_csv('D:\Practice\Book1.csv', chunksize=1)
# for p in chunker:
#     print(p)
#     print("---------------------------")

# -------------------------------------------------------



# Writing Data to Text Format

# df=pd.DataFrame(np.arange(9).reshape(3,3),
#                 columns=['c1','c2','c3'],
#                 index=['x',np.nan,'z'])


# df.to_csv('Created.csv')



# import sys
# df.to_csv(sys.stdout, sep='|')
# df.to_csv(sys.stdout, na_rep='NULL')

# ----------------------------------------------

# Working with Delimited Formats

# import csv
# f = open('Book1.csv')

# # reader = csv.reader(f)
# reader = csv.reader(f, delimiter='\n')




# for line in reader:
#     print(line)

# -----------------------------------------------------

# Reading Microsoft Excel Files

# xl=pd.ExcelFile(r"D:\Practice\ex.xlsx")

# df=pd.read_excel(xl,'Sheet1')
# print(df)




# df=pd.DataFrame(np.arange(9).reshape(3,3),
#                 columns=['c1','c2','c3'],
#                 index=['x','y','z'])

# df.to_excel('createdxl.xlsx')




# writer = pd.ExcelWriter('D:\Practice\ex.xlsx')
# df.to_excel(writer, 'Sheet1')
# writer.save()



# 00000000000000000000000000000000000000000000000000000000000000

# CHAPTER 7
# Data Cleaning and Preparation

# 7.1 Handling Missing Data

# # Filtering Out Missing Data
# #

# data= pd.Series([1,np.nan,2,3])
# print(data)

# # data= data.dropna()
# # print(data)
# print( data[data.notnull()])


# ---------------------------------------

# dAtaFrame 

# df=pd.DataFrame([[1,2,3,4],
#                  [5,np.nan,7,8],
#                  [12,34,23,8],
#                  [np.nan,np.nan,np.nan,np.nan]])
# print(df)

# df=df.dropna()
# print(df)

# df=df.dropna(axis=1,thresh=3) # minimum 3 NonNulls hai to mat karo

# print(df)

# -----------------------------------------------
# Filling In Missing Data



# df=pd.DataFrame([[1,2,3,4],
#                  [5,np.nan,7,8],
#                  [12,np.nan,23,8],
#                  [np.nan,np.nan,np.nan,np.nan]])
# print(df)

# df=df.fillna(0)
# print(df)
# df=df.fillna({0:'1stcolNan',1:'2ndcolNan'})
# print(df)


# df= df.fillna(method='ffill')
# print(df)
# df= df.fillna(method='ffill', limit=1)  #consecutive 1 only 
# print(df)


# ---------------------------------------------------------



# 7.2 Data Transformation

# Removing Duplicates

# df = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
#                      'k2': [1, 1, 2, 3, 3, 4, 4]})

# print(df)

# print(df.duplicated())
# # df=df.drop_duplicates()
# # print(df)
# df=df.drop_duplicates(['k1'])
# # print(df)
# df=df.drop_duplicates(['k1','k2'],keep='last')
# print(df)

# ---------------------------------------------------

# Transforming Data Using a Function or Mapping


# df = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
#                               'Pastrami', 'corned beef', 'Bacon',
#                               'pastrami', 'honey ham', 'nova lox'],
#                       'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})


# meat_to_animal = {
#  'bacon': 'pig',
#  'pulled pork': 'pig',
#  'pastrami': 'cow',
#  'corned beef': 'cow',
#  'honey ham': 'pig',
#  'nova lox': 'salmon'
# }

# loweredfood= df['food'].str.lower()

# print(loweredfood)

# df['animal'] = loweredfood.map(meat_to_animal)
# print(df)


# -------------------------------------------
# Replacing Values

# s=pd.Series([1,2,3,4,5])
# print(s)
# # s=s.replace(2,np.nan)
# # print(s)

# s=s.replace({1: np.nan, 2: 0})

# print(s)


# -----------------------------------------------
# Renaming Axis Indexes


# df = pd.DataFrame(np.arange(12).reshape((3, 4)),
#                     index=['Ohio', 'Colorado', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])

# print(df)


# transform = lambda x: x[:4]

# df.index=df.index.map(transform)

# print(df)

# df=df.rename(index=str.title, columns = str.upper)
# print(df)

# df=df.rename(index={'Ohio': 'Blank'},columns={'THREE':'Blank'})
# print(df)



# ------------------------------------------------------


# Discretization and Binning


# cut: equal intervals but unequal no of elem

# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 35, 60, 100]

# cats = pd.cut(ages, bins)
# print(cats)
# print(cats.categories)

# [(18, 25] < (25, 35] < (35, 60] < (60, 100]]

# print(pd.value_counts(cats))


# if you want to include upper value and not lower

# cats  =pd.cut(ages, bins, right=False)

# print(cats)

#  [[18, 25) < [25, 35) < [35, 60) < [60, 100)]

# groups = ['Youth', 'YoungAdult', 
#                'MiddleAged', 'Senior']

# cats  = pd.cut(ages, bins, labels=groups)

# print(cats)

# ------------------------------------

# qcut : unequal intervals but equal no. of elem

# data = np.random.randn(8)
# cats= pd.qcut(data,4)

# print(cats)
# print(cats.value_counts())


# ---------------------------------------------


# Detecting and Filtering Outliers

# df = pd.DataFrame(np.random.randn(3,3),index=['x','y','z'],
#                                 columns=['a','b','c'])
# print(df)
# df=(df[(np.abs(df)>0.5)])
# print(df)
# df=df.loc[:,(np.abs(df) > 1).any(axis=0)]
# print(df)
# df=df.loc[(np.abs(df) > 1).any(axis =1),:]
# print(df)


# print(np.sign(df))

# -------------------------------------------


# ---------------------------------------------

# 7.3 String Manipulation

# val= ' a , b , guido'
# print(val)

# print(val.split(','))   #list will be created
# for x in val.split(','):
#     print(x)
#     print()


# print(val.split(','))

# one,two ,three= val.split(',')
# print(one.strip())
# print(two.strip())


# val= ' a , b , guido'
# print('guido' in val)
# print( val.index(','))
# print(val.find(':'))
# print(val.index(':'))




# 00000000000000000000000000000000000000000000000000000000000000000000
# CHAPTER 8

# Data Wrangling: Join, Combine,
# and Reshape
# ---------------------------------------------------
# 8.1 Hierarchical Indexing


# df= pd.Series(np.random.randn(9),
#              index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
#                    ,[1, 2, 3, 1, 3, 1, 2, 2, 3]])

# print(df)
# print(df['b'])
# print(df.loc[['b', 'd']])




# print(df.loc[:, 2])
# For index ('a', 1), the associated value is approximately -1.662079.
# For index ('c', 2), the associated value is approximately 1.573635.
# For index ('d', 3), the associated value is approximately -0.170474.





# df=pd.DataFrame(np.arange(9).reshape(3,3),
#                 index= ['x','y','z'],
#                 columns =['a','b','c'])

# print(df)

# df=df.stack()
# print(df)
# df=df.unstack().stack()
# print(df)

# -------------------------------------------


# df = pd.DataFrame(np.arange(12).reshape((4, 3)),
#         index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
#         columns=[['Ohio', 'Ohio', 'Colorado'],
#         ['Green', 'Red', 'Green']])

# # print(df)

# df.index.names = ['key1', 'key2']
# df.columns.names = ['state', 'color']

# print(df)
# print(df['Ohio'])
# print(df.loc[('a', 1)])
# print(df.loc[(slice(None), 1), :])
# print(df['Ohio'])

# print(df.loc[:, (slice(None), 'Green')])


# --------------------------------------------------

# Reordering and Sorting Levels


# df= pd.Series(np.random.randn(9),
#              index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
#                    ,[1, 2, 3, 1, 3, 1, 2, 2, 3]])
# print(df)
# print('---------------------------')
# df.swaplevel()
# print(df)


# df=df.swaplevel().sort_index(level=1)
# print(df)

# df=df.swaplevel()

# -----------------------------------------------

# Summary Statistics by Level




# df= pd.Series([1,2,3,4,5,6,7,8,9],
#              index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
#                    ,[1, 2, 3, 1, 3, 1, 2, 2, 3]],
#                 )

# df.index.names = ['key1', 'key2']

# print(df)

# print(df.sum(level='key1',axis=0))


# -------------------------------------------------


# Indexing with a DataFrame’s columns


# df = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
#                        'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
#                        'd': [0, 1, 2, 0, 1, 2, 3]})

# print(df)

# df1=df.set_index(['c','d'],drop=False)
# print(df1)
# df1=df.set_index(['c','d'])
# print(df1)

# df1.reset_index()
# print(df1)



# ---------------------------------------------------

# 8.2 Combining and Merging Datasets

# df1=pd.DataFrame({'key':['b', 'b', 'a', 'c', 'a', 'a', 'b'],
#                   'data1':range(7)})
# df2=pd.DataFrame({'key':['a','b','d'],
#                   'data2':range(3)})


# print(df1)
# print(df2)
# print('----------------')

# print(pd.merge(df1,df2,how='inner'))
# print(pd.merge(df1,df2,how='outer'))
# print(pd.merge(df1,df2,how='left'))
# print(pd.merge(df1,df2,how='right'))

# -------------------------------------------


# df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
# 'data1': range(7)})
# df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
# 'data2': range(3)})


# print(df3)
# print(df4)

# print( pd.merge(df3, df4, left_on='lkey', right_on='rkey'))


# ---------------------------------------

# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
#       'data1': range(6)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
#       'data2': range(5)})

# print(df1)
# print(df2)

# print(pd.merge(df1, df2, on='key', how='left'))




# -----------------------------------------------------


# left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
#       'key2': ['one', 'two', 'one'],
#       'lval': [1, 2, 3]})
# right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
#       'key2': ['one', 'one', 'one', 'two'],
#       'rval': [4, 5, 6, 7]})



# print(left)
# print(right)

# print(pd.merge(left, right, on=['key1', 'key2'], how='outer'))


# ----------------------------------------------------------

# Merging on Index

# lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio',
# 'Nevada', 'Nevada'],
# 'key2': [2000, 2001, 2002, 2001, 2002],
#  'data': np.arange(5.)})

# righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
# index=[['Nevada', 'Nevada', 'Ohio', 'Ohio',
# 'Ohio', 'Ohio'],
# [2001, 2000, 2000, 2000, 2001, 2002]],
# columns=['event1', 'event2'])

# print(lefth)
# print(righth)

# print( pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True))


# -----------------------------------------------------



# left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
#                     index=['a', 'c', 'e'],
#                     columns=['Ohio', 'Nevada'])                 


# right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
#                     index=['b', 'c', 'd', 'e'],
#                     columns=['Missouri', 'Alabama'])
# print(left2)
# print(right2)

# print(left2.join(right2, how='outer'))
# print(left2.join(right2))

# ----------------------------------------------

# Concatenating Along an Axis

# s1 = pd.Series([0, 1], index=['a', 'b'])
# s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
# s3 = pd.Series([5, 6], index=['f', 'g'])

# print(pd.concat([s1, s2, s3]))

# print(pd.concat([s1, s2, s3], axis=1))

# codebasics video
# --------------------------------------------



# 8.3 Reshaping and Pivoting



# data = pd.DataFrame(np.arange(6).reshape((2, 3)),
#             index=pd.Index(['Ohio', 'Colorado'], name='state'),
#             columns=pd.Index(['one', 'two', 'three'],
#             name='number'))

# print(data)

# res =data.stack()
# print(res)
# print('-----------------')
# print(res.unstack())
# print(res.unstack(1))

# --------------------------------

# data = pd.DataFrame(np.arange(6).reshape((2, 3)),
#             index=pd.Index(['Ohio', 'Colorado'], name='state'),
#             columns=pd.Index(['one', 'two', 'three'],
#             name='number'))

# print(data)

# res =data.stack()
# print(res)

# print('-----------------')
# print(res.unstack('state'))
# # print(res.unstack(1))


# ------------------------------------------------------


# CHAPTER 9
# Plotting and Visualization

