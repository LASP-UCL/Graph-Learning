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
	ncols=int((n-1)*n/2)
	I=np.zeros(ncols)
	J=np.zeros(ncols)
	k=0
	for i in list((np.arange(n)+1))[1:]:

		I[k:k+(n-i)+1]=(np.arange(n)+1)[i-1:]
		k=k+(n-i+1)

	k=0
	for i in list((np.arange(n)+1))[1:]:
		J[k:k+(n-i)+1]=i-1
		k=k+n-i+1

	I=I.astype(int)
	J=J.astype(int)
	ys=list(I)+list(J)
	xs=list(np.arange(ncols)+1)+list(np.arange(ncols)+1)
	s_T=np.zeros((ncols,n))
	for i in range(len(ys)):
		s_T[(xs[i]-1),(ys[i]-1)]=1

	return s_T.T

def vector_form(W,n):
	w=W[np.triu_indices(n,1)]
	# the triangle-upper 
	return w
