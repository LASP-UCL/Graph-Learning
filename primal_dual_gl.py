import numpy as np
import random
from random import choice
import datetime
from matplotlib.pylab import *
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns
from synthetic_data import *
from scipy.optimize import minimize
from utils import sum_squareform, vector_form



class Primal_dual_gl():
	def __init__(self, node_num, Z, alpha, beta, w_0, d_0, gamma, eplison, iteration):
		self.node_num=node_num
		self.ncols=int(node_num*(node_num-1)/2)
		self.Z=Z
		self.z=vector_form(Z, node_num)
		self.alpha=alpha
		self.beta=beta
		self.W=np.zeros((node_num,node_num))
		self.w=w_0
		self.d=d_0
		self.S=sum_squareform(node_num)
		self.gamma=gamma
		self.eplison=eplison
		self.iteration=iteration
		self.y=None
		self.y_bar=None
		self.p=None
		self.p_bar=None
		self.q=None
		self.q_bar=None

	def run(self):
		for i in range(self.iteration):
			print('iteration', i)
			self.y=self.w-self.gamma*(2*self.beta*self.w+np.dot(self.S.T, self.d))
			self.y_bar=self.d+self.gamma*(np.dot(self.S, self.w))
			
			temp=self.y-2*self.gamma*self.z
			print('temp', temp)
			neg=np.where(temp<0)
			temp[neg]=0
			print('temp', temp)
			
			self.p=temp
			# if np.linalg.norm(self.y-2*self.gamma*self.z)>0:
			# 	self.p=self.y-2*self.gamma*self.z
			# else:
			# 	self.p=np.zeros(self.ncols)
			#p=np.max([0, self.y-2*self.gamma*self.z])
			self.p_bar=(self.y_bar-np.sqrt((self.y_bar)**2+4*self.alpha*self.gamma))/2.0
			self.q=self.p-self.gamma*(2*self.beta*self.p+np.dot(np.dot(self.S.T, self.S), self.p))
			self.q_bar=self.p_bar+self.gamma*(np.dot(self.S, self.p))


			w_i_1=self.w.copy()
			print('self.w', self.w)
			print('np.norm(w_i_1)', np.linalg.norm(w_i_1))
			print('self.y', self.y)
			print('self.p', self.p)
			self.w=w_i_1-self.y+self.p
			print('self.w', self.w)

			d_i_1=self.d.copy()
			self.d=d_i_1-self.y_bar+self.q_bar
			w_diff=np.linalg.norm(self.w-w_i_1)
			print('w_diff', w_diff)
			w_ratio=w_diff/np.linalg.norm(w_i_1)
			d_diff=np.linalg.norm(self.d-d_i_1)
			d_ratio=d_diff/np.linalg.norm(d_i_1)
			if (w_ratio<self.eplison) and (d_ratio<self.eplison):
				break 
			else:
				pass 
			index1=np.tril_indices(self.node_num, -1)
			index2=np.triu_indices(self.node_num,1)
			self.W[index1]=self.w
			self.W[index2]=self.w
		return self.w, self.W
