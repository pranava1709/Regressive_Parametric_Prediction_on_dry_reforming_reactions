from lin import lr
import pandas as pd 
import numpy as np 
import glob2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

lst_pred = []
lstaa = []

	


data = pd.read_csv('C:/Capstone_Results/dataset/RXNs at 800C1r.csv')
train,test = train_test_split(data,test_size = 0.30,random_state = None, shuffle = False)
xtrain = train.iloc[:,0:24]
ytrain = train.iloc[:,25:26]
xtest = test.iloc[:,0:24]
ytest = test.iloc[:,25:26]
	




mod = lr(noi = 1000,lr = 0.0000000000000001)

mod.weigh_bias(xtrain,ytrain)
predictions = mod.pred(xtest)
print(predictions)



for ii in range(0,predictions.shape[0]):
	predicted = predictions[ii][0]	
	lst_pred.append(predicted)
pre = np.array(lst_pred)
print(pre)
print(ytest)
sc = mod.metric(ytest,pre)
print(sc)
#sc = mod.metric(x,y)
score = max( 0 , 100*(1-mean_absolute_percentage_error(ytest,pre)))
print(score)

plt.title('EXPERIMENTAL DATA VS ML_MODEL_PREDICTIONS FOR DRY REFORMING')
plt.xlabel('TIME(in minutes)')
plt.ylabel('CARBON DIOXIDE PREDICTED')
time = xtest.iloc[:,1:2]

plt.plot(time,pre,label = 'Experimental')
plt.plot(time, ytest,label = 'Predicted')
plt.legend()

plt.show()
