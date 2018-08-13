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
from community import community_louvain
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns

def rbf_graph(node_num, dimension=2, threshold=0.75):
	positions=np.random.uniform(low=-0.5, high=0.5, size=(node_num, dimension))
	adj_matrix=rbf_kernel(positions, gamma=(1)/(2*(0.5)**2))
	adj_matrix[np.where(adj_matrix<threshold)]=0.0
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian

def er_graph(node_num, prob=0.2, seed=2018):
	graph=nx.erdos_renyi_graph(node_num, prob, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian

def ba_graph(node_num, seed=2018):
	graph=nx.barabasi_albert_graph(node_num, m=1, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)

	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian	


def find_eigenvalues_matrix(eigen_values):
	eigenvalues_matrix=np.diag(np.sort(eigen_values))
	return eigenvalues_matrix

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix


def generate_signal(signal_num, node_num, laplacian):
	mean=np.zeros(node_num)
	sigma_error=0.2
	pinv_lap=np.linalg.pinv(laplacian)
	cov=pinv_lap+sigma_error*np.identity(node_num)
	signals=np.random.multivariate_normal(mean, cov, size=signal_num)
	return signals

def find_corrlation_matrix(signals):
	corr_matrix=np.corrcoef(signals.T)
	return corr_matrix



