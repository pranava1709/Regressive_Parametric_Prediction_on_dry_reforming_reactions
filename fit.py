'''
This code is written by Mr. Seth Pranava. 
This is model training code
'''
from lin import lr
import pandas as pd 
import numpy as np 
import glob2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

lst_pred = []
lstaa = []

path = "C:/Capstone"
res = glob2.glob(path+'/*csv')


data = pd.read_csv('C:/Capstone_Results/dataset/RXNs at 800C1r.csv')
train,test = train_test_split(data,test_size = 0.30,random_state = None, shuffle = False)
xtrain = train.iloc[:,0:24]
ytrain = train.iloc[:,25:26]
xtest = test.iloc[:,0:24]
ytest = test.iloc[:,25:26]
	
#x= data.iloc[:,0:7]
#y = data.iloc[:,8:9]
#xtest = x.iloc[26:,0:8]




#mod = lr(noi = 1000,lr = 0.000001)
mod = lr(noi = 1000,lr = 0.0000000000000001)

mod.weigh_bias(xtrain,ytrain)
predictions = mod.pred(xtest)
print(predictions)
'''
'''
#sc = mod.metric(xtest,predictions)
#print(sc)
'''
'''
for ii in range(0,predictions.shape[0]):
	predicted = predictions[ii][0]	
	lst_pred.append(predicted)
pre = np.array(lst_pred)
print(pre)
print(ytest)
#sc = mod.metric(x,y)
score = max( 0 , 100*(1-mean_absolute_percentage_error(ytest,pre)))
print(score)
'''
#print(score)
#time = data.iloc[0:13,0:1]
#for aa in range(8,713,8):
#	lstaa.append(aa#)
#for aa in range(8,713,8):
#	lstaa.append(aa
'''
#plt.xlim()
#plt.ylim(0,100)
plt.title('EXPERIMENTAL DATA VS ML_MODEL_PREDICTIONS FOR DRY REFORMING')
plt.xlabel('TIME(in minutes)')
plt.ylabel('CARBON DIOXIDE PREDICTED')
time = xtest.iloc[:,1:2]

plt.plot(time,pre,label = 'Experimental')
plt.plot(time, ytest,label = 'Predicted')
plt.legend()
#plt.plot(time,pre)
#plt.plot(time, ytest)
#plt.scatter(ytest,pre)
#plt.plot(y[0:13],pre)

#plt.plot([70,100],[70,100])
plt.show()
'''
data = pd.read_csv('dataset.csv')
	
x= data.iloc[:,0:24]
y = data.iloc[:,24:25]



mod = lr(noi = 1000,lr = 0.0000000000000001)
mod.weigh_bias(x,y)
predictions = mod.pred(x)
for ii in range(0,predictions.shape[0]):
	predicted = predictions[ii][0]	
	lst_pred.append(predicted)
pre = np.array(lst_pred)
print(pre)
for aa in range(8,713,8):
	lstaa.append(aa)
plt.scatter(y,pre)
plt.title("SIMULATED VS PREDICTED")
plt.xlabel('SIMULATED')
plt.ylabel('PREDICTED METHANE CONVERSION')
plt.plot([0.9,1.0],[0.9,1.0])
#plt.plot(lstaa,y)
#plt.plot(lstaa,pre)

plt.show()
'''