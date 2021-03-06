## Graph learning and signal learning 
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
import datetime
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/test_results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=10
signal_num=10
error_sigma=0.5
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)

newpath=path+'error_sigma_%s'%(error_sigma)+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signal_error_list=[]
graph_error_list=[]
knn_graph_error_list=[]
knn_signal_error_list=[]
trace1=[]
trace2=[]
trace3=[]
trace4=[]

signal=X[0,:]

for i in np.arange(10):
	print('i',i)

	Z=euclidean_distances(signal.reshape(-1,1), squared=True)
	np.fill_diagonal(Z, 0)
	## Learn Graph
	alpha=0.1
	beta=1
	primal_gl=Primal_dual_gl(node_num, Z, alpha, beta)
	primal_adj, error=primal_gl.run(adj_matrix)
	laplacian=csgraph.laplacian(primal_adj, normed=False)
	## Learn signal
	gamma = 3
	G=graphs.Graph(primal_adj)
	G.compute_differential_operator()
	L = G.D.toarray()
	d = pyunlocbox.functions.dummy()
	r = pyunlocbox.functions.norm_l2(A=L, tight=False)
	f = pyunlocbox.functions.norm_l2(w=1, y=signal.copy(), lambda_=gamma)

	step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma, 2)
	solver = pyunlocbox.solvers.gradient_descent(step=step)
	x0 = signal.copy()
	prob2 = pyunlocbox.solvers.solve([r, f], solver=solver,
	                                x0=x0, rtol=0, maxit=2000)
	signal=prob2['sol']



	fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
	ax1.pcolor(adj_matrix, cmap='RdBu')
	ax1.set_title('real W')
	ax2.pcolor(primal_adj, cmap='RdBu')
	ax2.set_title('learned w')
	plt.show()


	tr1=np.trace(np.dot(signal.reshape(1,node_num), np.dot(laplacian, signal.reshape(1,node_num).T) ))
	tr2=np.trace(np.dot(signal.reshape(1,node_num), np.dot(knn_lap, signal.reshape(1,node_num).T)))
	trace1.extend([tr1])
	trace2.extend([tr2])
	real_signal=X
	tr3=np.trace(np.dot(real_signal, np.dot(laplacian, real_signal.T) ))
	tr4=np.trace(np.dot(real_signal, np.dot(knn_lap, real_signal.T)))
	trace3.extend([tr3])
	trace4.extend([tr4])



	signal_error=np.linalg.norm(signal-X[0,:])
	graph_error=np.linalg.norm(primal_adj-adj_matrix)
	signal_error_list.extend([signal_error])
	graph_error_list.extend([graph_error])



plt.plot(signal_error_list, label='GL')
plt.ylabel('Signal Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.savefig(newpath+'signal_error'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()


plt.plot(graph_error_list, label='GL')
plt.ylabel('Graph Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.savefig(newpath+'graph_error'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()



plt.plot(trace1, label='signal/lap')
plt.plot(trace2, label='signal/knn')
plt.plot(trace3, label='X/lap')
plt.plot(trace4, label='X/knn')
plt.legend(loc=1)
plt.show()


############################################ Results
## Real Graph and real signal
real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=X[0,:],node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.savefig(newpath+'real_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)

plt.show()


### Learned graph and learned signal
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.savefig(newpath+'learned_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)

plt.show()


##Plot w
plt.figure(figsize=(5,5))
plt.pcolor(adj_matrix, cmap='RdBu', vmin=np.min(adj_matrix), vmax=np.max(adj_matrix))
plt.colorbar()
plt.title('Real Adjacency Matrix')
plt.savefig(newpath+'real_w'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.pcolor(primal_adj, cmap='RdBu', vmin=np.min(adj_matrix), vmax=np.max(adj_matrix))
plt.colorbar()
plt.title('Learned Adjacency Matrix')
plt.savefig(newpath+'learned_w'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()






plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.show()


