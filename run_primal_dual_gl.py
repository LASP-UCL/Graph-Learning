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
from synthetic_data import *
from gl_sigrep import Gl_sigrep 
from primal_dual_gl import Primal_dual_gl 
from utils import vector_form, sum_squareform

node_num=50
signal_num=100
rbf_adj, rbf_lap=rbf_graph(node_num)
X_t=generate_signal(signal_num, node_num, rbf_lap)
X=X_t.T

Z=rbf_kernel(X)

iteration=100
alpha=0.012
beta=0.08
w_0=np.zeros(int((node_num-1)*node_num/2))
d_0=np.zeros(node_num)
#d=Sw
epsilon=10**(-6)
gamma=1+2*beta+np.sqrt(2*node_num-1)-5
gamma=

primal_gl=Primal_dual_gl(node_num, Z, alpha, beta, w_0, d_0, gamma, epsilon, iteration)

vector_adj, primal_adj=primal_gl.run()




# error=np.linalg.norm(output-signals)/signal_num

# lap_error=np.linalg.norm(rbf_lap-lap, 'fro')/signal_num

fig, (ax1, ax2)=plt.subplots(1,2, figsize=(8,4))
ax1.imshow(rbf_adj)
ax1.set_title('Ground Truth Laplacian')
ax2.imshow(primal_adj)
ax2.set_title('Learned Laplacian')
plt.show()