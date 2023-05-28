import numpy as numpy
import pandas as pd 
import numpy as np
import math
from matplotlib import pyplot as plt
import glob2
import os
from sklearn.metrics import mean_absolute_percentage_error
lsty = []
lstycap = []
lstw = []
lstb = []
mmaplst = []
pred = []
lst_pred = []

class lr:
	def __init__(self,noi,lr ):
		
		self.noi = noi
		self.lr = lr
		self.weights=None

	
	def weigh_bias(self,x,y):
		self.m = x.shape[0]
		x=pd.DataFrame(x)
		x1 = x.values
		y = y.values
		

		print(type(x))
		print(self.m)
		cost_lis = []
		
		self.bias = 0
		self.m1 = 0.5*len(y)

		self.weights =np.zeros(x1.shape[1])
		#self.weights = np.expand_dims(self.weights,axis = 1)

		
		for ii in range(self.noi):

			print(x.shape)
			print(x1.shape)
			print(self.weights.shape)
			self.ycap= np.dot(x1,self.weights.T)+self.bias
			self.residual  =self.ycap-y
			self.po = self.residual *self.residual 
			self.cost =  self.m1*np.sum(self.po)
			cost_lis.append(self.cost)
					#for jj in range (0,x.shape[1]):
			dw = np.dot(x1.T,(self.ycap-y))*2/x.shape[0]
			#print(dw)
			db = np.sum(self.ycap-y)/x.shape[0]
								#print(wg.shape)
			aa =self.lr*dw
			aa1 = self.lr*db
			self.weights= self.weights-aa.T
				#print(self.weights)

							
			self.bias  = self.bias -aa1
				#print(self.bias)
				
	def pred(self,x):
		x = x.values
				
		predictions= np.dot(x,self.weights.T)+ self.bias

		return predictions
	def metric(self,x,y):
		
		
		predictions = self.pred(x)
		print(predictions)
		for ii in range(0,predictions.shape[0]):
			predicted = predictions[ii][0]	
			lst_pred.append(predicted)
			print(lst_pred)
		aa = np.array(lst_pred)
		print(aa.shape)
		score = max( 0 , 100*(1-mean_absolute_percentage_error(y,aa)))
		return score
		
