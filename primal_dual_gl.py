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


def sum_squareform(n):
	#sum operator that find degree from upper triangle
	ncols=(n-1)*n/2
	I=np.zeros(ncols)
	J=np.zeros(ncols)
	k=0
	for i in list((np.arange(n)+1))[1:]:
		print('i',i)

		I[k:k+(n-i)+1]=(np.arange(n)+1)[i-1:]
		print('I',I)
		print('k', k)
		k=k+(n-i+1)
		print('k',k)

	k=0
	for i in list((np.arange(n)+1))[1:]:
		print('i',i)
		J[k:k+(n-i)+1]=i-1
		print('J',J)
		print('k',k)
		k=k+n-i+1
		print('k',k)

	I=I.astype(int)
	J=J.astype(int)
	ys=list(I)+list(J)
	xs=list(np.arange(ncols)+1)+list(np.arange(ncols)+1)
	s_T=np.zeros((ncols,n))
	for i in range(len(ys)):
		s_T[(xs[i]-1),(ys[i]-1)]=1

	print('s_T', s_T)
	return s_T.T

def vector_form(W,n):
	w=W[np.triu_indices(n,1)]
	# the triangle-upper 
	return w







class Primal_dual_gl():
	def __init__(self, node_num, Z, alpha, beta, w_0, d_0, gamma, eplison, iteration):
		self.node_num=node_num
		self.Z=Z
		self.z=vector_form(Z)
		self.alpha=alpha
		self.beta=beta
		self.w=w_0
		self.d=d_0
		self.S=sum_squareform(node_num)
		self.gamma=gamma
		self.eplison=eplison
		self.iteration=iteration

	def run(self):
		for i in range(iteration):
			y=self.w-self.gamma*(2*self.beta*self.w+np.dot(self.S.T, self.d))
			y_bar=self.d+self.gamma*(np.dot(self.S, self.w))
			p=np.max([0, y-2*self.gamma*self.z])
			p_bar=(y_bar-np.sqrt((y_bar)**2+4*self.alpha*self.gamma))/2.0
			q=p-self.gamma*(2*self.beta*p+np.dot(self.S.T,p))
			q_bar=p_bar+self.gamma*(np.dot(self.S, p))
			w_i_1=self.w
			self.w=self.w-y+p
			d_i_1=self.d
			self.d=self.d-y_bar+q_bar
			w_diff=np.linalg.norm(self.w-w_i_1)/np.linalg.norm(w_i_1)
			d_diff=np.linalg.norm(self.d-d_i_1)/np.linalg.norm(d_i_1)
			if (w_diff<self.eplison) and (d_diff<self.eplison):
				break 
			else:
				pass 
		return self.w
