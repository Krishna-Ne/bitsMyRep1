#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#trgfile = "C:/Users/krishna/Desktop/nk1252/ExecutivePoint/BITS/Data #Analytics/water prj/xx.csv"
"salary.csv"
#dataset = pd.read_csv(file,delimiter = ',', names=names)

trgnames = ['Yrs','Skill','Role','Salary']
trgdataset = pd.read_csv("salary.csv", names=trgnames,usecols=[0, 1,2,3], nrows = 30) 

# shape
#print(trgdataset.shape)
print(trgdataset.head(30)) 
# box and whisker plots
#trgdataset.plot(kind='box', subplots=True, layout=(16,16), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

trgarray = trgdataset.values
X = trgarray[:,0:-1]
Y = trgarray[:,3]
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.01, random_state=3)


#print(trgdataset.groupby('Role'))
#trgdataset.hist()
#plt.show()

#from pandas.plotting import scatter_matrix
# multivarite plot
#scatter_matrix(trgdataset)


print("stp2 for RandomForest ")
model = RandomForestClassifier(n_estimators=100)

#print("stp2 for KNN with testdata")
#model = KNeighborsClassifier()

#print("stp2 for  LinearSVC with testdata")
#model = SVC(kernel='linear', C = 1.0)

#print("stp2 for  Poly SVC with testdata")
#model = SVC(kernel='poly', gamma = 'scale',  degree=8)

#print("stp2 for  Gausian SVC with testdata")
#model = SVC(kernel='rbf', gamma = 'scale')

from sklearn import preprocessing  
lab_enc = preprocessing.LabelEncoder()
Y_encoded = lab_enc.fit_transform(Y)

model.fit(X,Y_encoded)

print("stp2 model with my test data")
testfile = "testsalary.csv"

tnames = ['Yrs','Skill','Role','Salary']

testdataset = pd.read_csv(testfile,names=tnames, usecols=[0, 1,2], nrows = 1)
testarray = testdataset.values
mytest = testarray[:, 0:3]
print (mytest)
predictions = model.predict(mytest )

print (mytest, "prediction with my test data;", predictions)

