
# -*- coding: utf-8 -*-

'''
LICENSE: BSD 2-Clause

Summer 2020
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, time
os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import networkx as nx
import random as rn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import numpy as np
from collections import OrderedDict
from scipy.sparse import load_npz, save_npz
import tqdm
from sklearn.manifold import TSNE
import pickle
from sklearn import metrics
from multiprocessing import Pool
from functools import partial
import copy
from matplotlib.backends.backend_pdf import PdfPages

from supervised_models import SupervisedGraphsage
from models_util import SAGEInfo
from minibatch import MinibatchIterator
from neigh_samplers import UniformNeighborSampler

tf.compat.v1.disable_eager_execution()
cut_off  = 0.5
glo_seed = 123


class Hypo_Gen():
	'''
	Temporal Relationship Prediction
	'''

	def __init__(self, current_year, verbose):
		
		self.verbose = verbose
		self.current_year = current_year
		self.globi = 0

		rn.seed(glo_seed)
		self.rng = np.random.RandomState(seed=glo_seed)
		np.random.seed(glo_seed)
		tf.compat.v1.set_random_seed(glo_seed)
		rn.seed(glo_seed)
		self.rng = np.random.RandomState(seed=glo_seed)
		np.random.seed(glo_seed)



	def get_neg_assume(self, item, full_ids):
		rez = []
		key, value = item
		diff = list(full_ids - value - {key})
		if len(diff) > self.max_neg_per_node:
			neg = self.rng.choice(diff, self.max_neg_per_node, replace=False)
		else:
			neg = diff

		return key, neg

	def sample_neighbors(self, new_node_id_map):
		
		node_count = len(self.G.nodes())

		node_neighbs = []
		for i in range(node_count):
			neighbs = [n for n in self.G.neighbors(i)]
			if len(neighbs) > self.max_degree:
				neighbs = rn.sample(neighbs, self.max_degree)
			elif len(neighbs) < self.max_degree and len(neighbs) >  0:
				neighbs = rn.choices(neighbs, k=self.max_degree)
			else:
				if i in new_node_id_map:
					neighbs = new_node_id_map[i]
					if len(neighbs) > self.max_degree:
						neighbs = rn.sample(neighbs, self.max_degree)
					elif len(neighbs) < self.max_degree and len(neighbs) >  0:
						neighbs = rn.choices(neighbs, k=self.max_degree)
					else:
						neighbs = [i]*self.max_degree
				else:
					neighbs = [i]*self.max_degree			   
 
			node_neighbs.append(neighbs)

		return np.array(node_neighbs)

	def saveLoad(self, opt, variables, year, data_folder='corona_data'):
		global calc
		dump_folder = 'dump_{}'.format(data_folder)
		if opt == "save":
			
			if not os.path.exists(dump_folder):
				os.mkdir(dump_folder)

			f = open("{}/HPcache_{}".format(dump_folder, year), 'wb')
			pickle.dump(variables, f, 2)
			f.close
			print ('data saved')
			return None
		elif opt == "load":

			try:
				f = open("{}/HPcache_{}".format(dump_folder,year), 'rb')
				variables = pickle.load(f)
				return variables
			except:
				print("Error in loading data")
				return None
		else:
			print ('Invalid saveLoad option')
			return None

	def setup_data(self, test_type='eval', data_folder='corona_data'):

		edge_list = []
		id_map, y_map = {}, {}
		fut_set = OrderedDict() 
		data = OrderedDict()
		nodes = set()
		_id, y_id = 0, 1
		train_mask = set()
		pair_id, mask_id = 0, 0
		node_neg = []
		node_neighbs = []
		pair_tracker = {}
		corona = set()
		year_neg_adj, year_node_neighbs = [], []
		nodes = set()
		year_mask = {}
		acp_opt = set(['none', 'any','full', 'all', 'new', 'neg', 'eval'])
		new_nodes = []
		test_mask = {test_set: set() for test_set in acp_opt}
		neg_test_mask = {test_set: set() for test_set in acp_opt}

		LoadedVariables =  self.saveLoad("load", None, self.end_year, data_folder)
		if LoadedVariables is not None:
			self.num_year_window, self.id_map, self.adj, self.seq, self.max_id,  data, masks, y_map, new_nodes = LoadedVariables
			return data, masks, y_map, new_nodes

		years = range(self.start_year, self.end_year + self.year_interval + 1, self.year_interval)

		window_id = 0
		self.num_year_window = len(years) -1


		for i in range(len(years)):
			try:
				for line in open("{}/id_maps/id_map_text_{}".format(data_folder, years[i])):
					pid = line.rstrip('\n') 
					try:
						if pid not in id_map:
							id_map[pid] = _id
							nodes.add(_id)
							_id += 1					
						full_ids.add(id_map[pid])
					except:
						pass
			except:
				pass

		next_cur_fut_conn = {}
		pool = Pool(15)
		for y in range(len(years[:-1])):
			year = years[y]
			print(year)
			full_ids = set()
			new_node_dic = {}
			cur_fut_conn = next_cur_fut_conn
			next_cur_fut_conn = {}
			edge_list =[]
			test_nodes = set([])

			if year == self.start_year:
				for line in open('{}/graph/tgraph_{}.edgelist'.format(data_folder, year)):
					content = line.rstrip('\n')
					s_pid, t_pid, _ = content.split(" ")
					edge_list.append([id_map[s_pid] , id_map[t_pid]])
					if id_map[s_pid] in cur_fut_conn:
						cur_fut_conn[id_map[s_pid]].add(id_map[t_pid])
					else:
						cur_fut_conn[id_map[s_pid]] = {id_map[t_pid]}

					if id_map[t_pid] in cur_fut_conn:
						cur_fut_conn[id_map[t_pid]].add(id_map[s_pid])
					else:
						cur_fut_conn[id_map[t_pid]] = {id_map[s_pid]}
				self.G = nx.Graph(edge_list)

			else:
				self.G = self.G_future

			edge_list_future =[]
			try:
				for line in open('{}/graph/tgraph_{}.edgelist'.format(data_folder, years[y+1])):
					content = line.rstrip('\n')
					s_pid, t_pid, _ = content.split(" ")
					edge_list_future.append([id_map[s_pid] , id_map[t_pid]])
					if id_map[s_pid] in next_cur_fut_conn:
						next_cur_fut_conn[id_map[s_pid]].add(id_map[t_pid])
					else:
						next_cur_fut_conn[id_map[s_pid]] = {id_map[t_pid]}

					if id_map[t_pid] in next_cur_fut_conn:
						next_cur_fut_conn[id_map[t_pid]].add(id_map[s_pid])
					else:
						next_cur_fut_conn[id_map[t_pid]] = {id_map[s_pid]}

				
				self.G_future = nx.Graph(edge_list_future)
			except:
				self.G_future = self.G

			graph_nodes = set(self.G.nodes())
			if year == self.end_year:
				self.max_neg_per_node = 40
			else:
				self.max_neg_per_node = 20

			cur_nodes = self.G.nodes()
			for node in cur_nodes:
				year_pid = "{}_{}".format(node, year)
				y_map[year_pid] = y_id
				y_id += 1

			node_difference = set(self.G_future.nodes()) - set(cur_nodes)
			new_node_id_map = {}
			node_icd = set()
			for node in node_difference:
				neighbors = set([ n for n in self.G_future.neighbors(node)])
				avail_in_graph = graph_nodes & neighbors
				if len(avail_in_graph)> 0:
					year_pid = "{}_{}".format(node, year)
					new_node_dic[year_pid] = [y_map["{}_{}".format(n, year)] for n in avail_in_graph]
					new_node_id_map[node] = list(avail_in_graph)
					y_map[year_pid] = y_id
					node_icd.add(node)
					y_id += 1
			new_nodes.append(new_node_dic)

			for e in self.G_future.edges():
				s, t = e
				if not self.G.has_edge(s,t):
					if s in cur_fut_conn:
						cur_fut_conn[s].add(t)
					else:
						cur_fut_conn[s] = {t}

					if t in cur_fut_conn:
						cur_fut_conn[t].add(s)
					else:
						cur_fut_conn[t] = {s}

					frwd, bck = "{}_{}".format(s, t), "{}_{}".format(t, s)
					
					if frwd not in pair_tracker and bck not in pair_tracker:
						pair_tracker[frwd] = pair_id
						cur_pair_id = pair_id
						
						data[cur_pair_id] = [s, t, window_id]
						pair_id += 1
						added = True
					elif frwd in pair_tracker:
						cur_pair_id = pair_tracker[frwd]
						data[cur_pair_id][-2] = window_id

						added = False
					else:
						cur_pair_id = pair_tracker[bck]
						data[cur_pair_id][-2] = window_id


						added = False
					

					if  year == self.end_year:
						if added == False:
							if cur_pair_id in train_mask:
								train_mask.remove(cur_pair_id)
						if test_type != 'eval':
							if (s in graph_nodes) and  (t in graph_nodes):
								test_mask['all'].add(cur_pair_id)
							if (s in graph_nodes) or  (t in graph_nodes):
								test_mask['any'].add(cur_pair_id)
							if (s in node_difference) or  (t in node_difference):
								test_mask['new'].add(cur_pair_id)
							test_mask['full'].add(cur_pair_id)
							if 'C000657245' in id_map and (s == id_map['C000657245'] or t == id_map['C000657245']):
								corona.add(cur_pair_id)
						else:
							new_eval_pairs = open("{}/eval_ids.txt".format(data_folder), 'rU').read().split('\n')[:-1]
							for node_pair in new_eval_pairs:
								es_pid, et_pid, yr = node_pair.split(" ")
								s, t = id_map[es_pid], id_map[et_pid]
								
								frwd, bck = "{}_{}".format(s, t), "{}_{}".format(t, s)
								
								if frwd not in pair_tracker and bck not in pair_tracker:
									pair_tracker[frwd] = pair_id
									
									if int(yr) <= year+self.year_interval:
										data[pair_id] = [s, t, window_id]
									else:
										data[pair_id] = [s, t, 100]
									test_mask['eval'].add(pair_id)
									pair_id += 1
								elif frwd in pair_tracker:
									test_mask['eval'].add(pair_tracker[frwd])
									if cur_pair_id in train_mask:
											train_mask.remove(cur_pair_id)
								else:
									test_mask['eval'].add(pair_tracker[bck])
									if cur_pair_id in train_mask:
										train_mask.remove(cur_pair_id)

							cur_fut_conn = {}
							break
					else:
						if (s in graph_nodes) and  (t in graph_nodes):
							train_mask.add(cur_pair_id)
							mask_id += 1
					
			self.max_id = _id
			edge_list =[]
			if year == self.end_year:
				full_ids = set(self.G_future.nodes())
			else:
				full_ids = set(self.G.nodes())
			order = []
			node_pair_list = list(cur_fut_conn.items())
			u_set = pool.map(partial(self.get_neg_assume, full_ids=full_ids), node_pair_list)
			for item in u_set:
				for x in item[1]:

					frwd = "{}_{}".format(item[0], x)
					bck = "{}_{}".format(x, item[0])
					if frwd not in pair_tracker and bck not in pair_tracker:
						pair_tracker[frwd] = pair_id
						cur_pair_id = pair_id
						data[cur_pair_id] = [item[0], x, 100]
						pair_id += 1
						added = True
					elif frwd in pair_tracker:
						cur_pair_id = pair_tracker[frwd]
						added = False
					else:
						cur_pair_id = pair_tracker[bck]

						added = False

					if added == False:
						if cur_pair_id in train_mask:
							train_mask.remove(cur_pair_id)
						
					if year == self.end_year:
						if (item[0] in graph_nodes) and  (x in graph_nodes):
							neg_test_mask['all'].add(cur_pair_id)
						if (item[0] in graph_nodes) or  (x in graph_nodes):
							neg_test_mask['any'].add(cur_pair_id)
						if (item[0] not in graph_nodes) or  (x not in graph_nodes):
							neg_test_mask['new'].add(cur_pair_id)

						neg_test_mask['full'].add(cur_pair_id)
					else:
						train_mask.add(cur_pair_id)

				order.append(item[0])

			self.G.add_nodes_from(set(range(_id)) - set(self.G.nodes))
			year_node_neighbs.append(self.sample_neighbors(new_node_id_map))
			window_id += 1

		pool.close()
		pool.join()
		self.id_map = id_map
		self.adj = np.array(year_node_neighbs)
		self.seq = np.transpose(self.create_series_data(nodes, y_map, data_folder))

		data,  masks = np.vstack(data.values()), [list(train_mask), test_mask, neg_test_mask, corona]
		variables = [self.num_year_window, self.id_map, self.adj, self.seq, self.max_id,  data, masks, y_map, new_nodes]
		self.saveLoad( "save", variables, self.end_year, data_folder)
		del variables

		return data, masks, y_map, new_nodes



	def create_series_data(self, nodes, id_map, data_folder):
		nodes = list(nodes)
		year_winds = range(self.start_year, self.end_year + 1, self.year_interval)

		seq = np.zeros((len(nodes), len(year_winds)), dtype=np.int32)
		for i in range(len(year_winds)):
			for j in range(len(nodes)):
				pid = "{}_{}".format(nodes[j], year_winds[i]) 
				if pid in id_map:
					seq[j,i] = id_map[pid]
		return seq



	def get_node_context_attribute(self, id_map, feature_dim):

		context_arr = np.zeros((len(id_map)+1, feature_dim))
		years = range(self.start_year, self.end_year + 1, self.year_interval)

		for year in years:
			node_context = load_npz("{}/emb/matrices/lsi/context_lsi_{}_emb.npz".format(self.FLAGS.data_folder, year)).toarray()
			extra_content_file = "{}/text_content/extra_context_{}".format(self.FLAGS.data_folder, year)
			
			for line in open(extra_content_file):
				context = line.rstrip('\n') 
				pid, text = context.split(" ")
				indx = text.split(",")
				o_pid = "{}_{}".format(self.id_map[pid], year)
				if len(indx) > 0:
					context_arr[id_map[o_pid]] = node_context[np.array(indx).astype(np.int32)-1, :].mean(0)

		return context_arr

	def calc_f1(self, y_true, y_pred, is_test=False):

		y_pred_ = np.squeeze((y_pred >= cut_off),-1)
		y_true = (y_true <= self.num_year_window).astype(np.int32)
		if is_test ==  True:
			labels = y_pred_ + (3 * y_true)

			indx2 = np.where(labels == 3)[0]	#observed but negative pred
			indx3 = np.where(labels == 1)[0]	#not observed but postive pred
			indx4 = np.where(labels == 4)[0]  #observed and positive pred

			tpr = self.safe_div(len(indx4),(len(indx4) + len(indx2)))
			fnr = self.safe_div(len(indx2),(len(indx4) + len(indx2)))

		y_pred_ = y_pred_.astype(np.float32)
		y_true = y_true.astype(np.float32)

		f1_mac = metrics.f1_score(y_true, y_pred_, average="macro")
		lrap = metrics.label_ranking_average_precision_score(y_true.reshape([1,-1]), y_pred.reshape([1,-1]))
		rec = metrics.recall_score(y_true, y_pred_, 'macro')
		f1_pu = (rec ** 2)/((y_pred_).sum()/y_pred_.shape[0])

		if is_test == False:
			return f1_mac,  lrap, None, None, f1_pu
		else:
			f1_bi = metrics.f1_score(y_true, y_pred_, average="binary")
			auc = metrics.roc_auc_score(y_true, y_pred)
			return f1_mac, lrap, f1_bi, auc, f1_pu

	# Define model evaluation function
	def evaluate(self, sess, model, minibatch_iter, size=None):
		t_test = time.time()
		feed_dict_val, labels = minibatch_iter.val_feed_dict(size)
		node_outs_val = sess.run([model.preds, model.loss], 
							feed_dict=feed_dict_val)
		f1, lrap, _, _, f1_pu   = self.calc_f1(labels, node_outs_val[0])
		
		return node_outs_val[1], f1, lrap, f1_pu, (time.time() - t_test)

	def analyze_paths2(self, K, max_len, edge, y_pred=None, test_set=None):
		sorted_idx = np.where(y_pred >= cut_off)[0]
		pred = y_pred[sorted_idx]
		sorted_idx = sorted_idx[np.argsort(pred)[::-1]]
		sorted_pred = np.sort(pred)[::-1]
		pos = edge[sorted_idx]
		edge_list = []
		id2id = [k for k, _ in sorted(self.id_map.items(), key=lambda item: item[1])]
		for line in open('{}/graph/tgraph_{}.edgelist'.format(self.FLAGS.data_folder,  self.end_year)):
			content = line.rstrip('\n')
			s_pid, t_pid, _ = content.split(" ")
			edge_list.append([s_pid , t_pid])

		G = nx.Graph(edge_list)

		dist = {}
		keys = {}
		comb = {}
		output_files = [open("output/{}_{}_{}.txt".format(self.cid, test_set, i), "w") for i in range(max_len)]
		count = [0 for _ in range(max_len)]
		i_index, n_index, key_types = 1, 2, ["keylist","keylist2.txt"]
		key_dics = {}

		for i in range(len(key_types)):
			for line in open("key_data/{}".format(key_types[i])):
				row = line.rstrip('\n') 
				try:
					items = row.split("\t") 
					if len(items) >= 4:
						if items[i_index] not in key_dics:
							key_dics[items[i_index]] = {items[n_index].lower()}
				except:
					pass

		# output_file.write("================Top K Positive prediction===============\n")
		for i  in range(pos.shape[0]):
			try:
				sh_path = nx.bidirectional_shortest_path(G, id2id[pos[i,0]], id2id[pos[i,1]])
				if count[len(sh_path) - 1] <= K:
					output_files[len(sh_path) - 1].write("Shortest path from {}({}) <--> {}({}) ---{} -- {}\n".format(key_dics[id2id[pos[i,0]]], id2id[pos[i,0]], key_dics[id2id[pos[i,1]]], id2id[pos[i,1]],i, sorted_pred[i]))
					for item in sh_path:
						output_files[len(sh_path) - 1].write("\t - {}\n".format(key_dics[item]))
					count[len(sh_path) - 1] += 1
			except Exception as e:
				# print(e)
				output_files[-1].write("No Paths found for {} <--> {}-- {} -- {}\n\n".format(key_dics[id2id[pos[i,0]]], key_dics[id2id[pos[i,1]]], i, sorted_pred[i]))

		for output_file in output_files:
			output_file.close()

	def pltcolor(self,lst):
		cols=[]
		for l in lst:
			if l == 0 :
				cols.append('red')
			elif l == 4:
				cols.append('blue')
			elif l == 3:
				cols.append('yellow')
			elif l == 1:
				cols.append('green')
		return cols


	def safe_div(self, n, d, ret_val=0):
		return n / d if d else ret_val

	def get_emb_and_plot(self, sess, model, minibatch_iter, size, test=False, test_set='full', full_eval=False):
		finished = False
		full_eval = True
		labels = []
		y_pred = []
		mu = []
		emb = []
		edge_link = []
		iter_num = 0
		pdf = PdfPages('output/{}_{}_{}.pdf'.format(self.FLAGS.data_folder,self.cid, test_set))
		corona = minibatch_iter.set_test_data(test_set, True)
		while not finished:
			feed_dict_val, batch_labels, finished, link,_  = minibatch_iter.incremental_val_feed_dict(size, iter_num, test=test)
			node_outs_val = sess.run([model.h_t, model.all_preds], 
							 feed_dict=feed_dict_val)
			labels.append(batch_labels)
			y_pred.append(node_outs_val[1])

			if full_eval:
				emb.append(node_outs_val[0])
				mu.append(emb[-1][-1, :, :])
				edge_link.append(link)
			else:
				mu.append(node_outs_val[0][-1, :, :])
			
			iter_num += 1
		
		y_pred = np.squeeze(np.concatenate(y_pred, 1))
		labels = np.hstack(labels)
		if full_eval:
			emb = np.concatenate(emb, 1)
			edge_link = np.vstack(edge_link)
			if test_set == 'full' and self.FLAGS.data_folder=="corona_data":
				self.analyze_paths2(100, 5, edge_link[corona,:], y_pred[-1,corona], test_set)
		mu = np.vstack(mu)


		y_tmp = copy.deepcopy(y_pred[-1])
		y_tmp = (y_tmp >= cut_off).astype(np.float32)
		labels = (labels <= self.num_year_window).astype(np.float32)
		labels = y_tmp + (3 * labels)

		indx = np.where(labels == 0)[0]		#not observed and negative pred
		indx2 = np.where(labels == 3)[0]	#observed but negative pred
		indx3 = np.where(labels == 1)[0]	#not observed but postive pred
		indx4 = np.where(labels == 4)[0]  #observed and positive pred

		indx = np.hstack((indx[:200],indx2[:200], indx3[:200], indx4[:200]))
		
		labels = labels[indx]
		mu = mu[indx]
		color = self.pltcolor(labels)

		fig = plt.figure(constrained_layout=True)
		spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
		X = TSNE(n_components=2).fit_transform(mu)
		ax = fig.add_subplot(spec2[0, 0])
		ax.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.Spectral)
		plt.title("Pair embedding scatter plot")

		pdf.savefig(fig)
		pdf.close()
		

	def incremental_evaluate(self, sess, model, minibatch_iter, size, test=False, test_set='full'):
		t_test = time.time()
		finished = False
		losses = []
		val_preds = []
		labels = []
		iter_num = 0
		finished = False

		minibatch_iter.set_test_data(test_set)

		while not finished:
			feed_dict_val, batch_labels, finished, _, _  = minibatch_iter.incremental_val_feed_dict(size, iter_num, test=test)
			node_outs_val = sess.run([model.preds, model.loss, model.prior], 
							 feed_dict=feed_dict_val)
			val_preds.append(node_outs_val[0])
			labels.append(batch_labels)
			losses.append(node_outs_val[1])
			iter_num += 1
		val_preds = np.vstack(val_preds)
		labels = np.hstack(labels)
		if test:
			print("Prior is : {}".format(node_outs_val[-1]))
		scores = self.calc_f1(labels, val_preds, test)

		return np.mean(losses), scores[0], scores[1], scores[2], scores[3], scores[4], (time.time() - t_test)

	def construct_placeholders(self, num_year_window):
		# Define placeholders
		placeholders = {
			'labels' : tf.compat.v1.placeholder(tf.int32, shape=(None,), name='labels'),
			'batch1' : tf.compat.v1.placeholder(tf.int32, shape=(None,), name='batch1'),
			'batch2' : tf.compat.v1.placeholder(tf.int32, shape=(None,), name='batch2'),
			'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
			'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
			'is_train' : tf.compat.v1.placeholder(tf.bool, name='is_train'),
		}
		return placeholders



	def train_test(self,FLAGS, verbose=False):

		num_classes = 1
		self.max_degree = FLAGS.max_degree

		if FLAGS.data_folder == "corona_data":
			self.year_interval = 5
			self.start_year = 1985
			self.end_year = 2015
		else:
			self.year_interval = 10
			self.start_year = 1969
			self.end_year = 2009

		feature_dim = 300
		self.cid = ('{0}_{1}_{2}_{3}_{4}_{5}'.format(
			FLAGS.data_folder,
			FLAGS.model,
			FLAGS.epochs,
			FLAGS.learning_rate,
			self.end_year,
			time.time()))
		data, masks, id_map, new_nodes= self.setup_data(FLAGS.test_type, FLAGS.data_folder)
		
		labels = data[:, 2]
		self.FLAGS= FLAGS

		years = range(self.start_year, self.end_year + 1, self.year_interval)

		print(years)

		features = np.zeros((len(id_map)+1, feature_dim))
		for year in years:
			tmp_features = load_npz("{}/emb/matrices/lsi/node_lsi_{}_emb.npz".format(self.FLAGS.data_folder, year)).toarray()
			ids = map(lambda x: id_map["{}_{}".format(self.id_map[x], year)], open("{}/id_maps/id_map_text_{}".format(self.FLAGS.data_folder, year)).read().split("\n")[:-1])
			features[list(ids),:] = tmp_features

		context_features = self.get_node_context_attribute(id_map, feature_dim)

		features = np.add(features, context_features)
		for i in range(len(new_nodes)):
			for node, neighbors in new_nodes[i].items():
				features[id_map[node]] = features[neighbors[:20]].mean()

		data = data[:, :2]

		placeholders = self.construct_placeholders(len(years)+1)
		minibatch = MinibatchIterator( data, labels, masks,
				placeholders, 
				self.seq.shape[0],
				max_id=self.max_id,
				batch_size=self.FLAGS.batch_size,
				max_degree=self.FLAGS.max_degree) 
				

		adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=self.adj.shape)
		adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

		seq_info_ph = tf.compat.v1.placeholder(tf.int32, shape=self.seq.shape)
		seq_info = tf.Variable(seq_info_ph, trainable=False, name="seq_info")

		if self.FLAGS.model == 'graphsage_mean':
			# Create model
			sampler = UniformNeighborSampler(adj_info)
			if self.FLAGS.samples_3 != 0:
				layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
									SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2),
									SAGEInfo("node", sampler, self.FLAGS.samples_3, self.FLAGS.dim_2)]
			elif self.FLAGS.samples_2 != 0:
				layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
									SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]
			else:
				layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]

			model = SupervisedGraphsage(labels, self.seq.shape[0], placeholders, 
										 features,
										 seq_info,
										 risk_type=self.FLAGS.risk_type,
										 layer_infos=layer_infos, 
										 model_size=self.FLAGS.model_size,
										 sigmoid_loss = self.FLAGS.sigmoid,
										 identity_dim = self.FLAGS.identity_dim,
										 logging=True)
		elif self.FLAGS.model == 'gcn':
			# Create model
			sampler = UniformNeighborSampler(adj_info)
			layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2*self.FLAGS.dim_1),
								SAGEInfo("node", sampler, self.FLAGS.samples_2, 2*self.FLAGS.dim_2)]

			model = SupervisedGraphsage(labels, self.seq.shape[0], placeholders, 
										 features,
										 seq_info,
										 risk_type=self.FLAGS.risk_type,
										 layer_infos=layer_infos, 
										 aggregator_type="gcn",
										 model_size=self.FLAGS.model_size,
										 concat=False,
										 sigmoid_loss = self.FLAGS.sigmoid,
										 identity_dim = self.FLAGS.identity_dim,
										 logging=True)

		elif self.FLAGS.model == 'graphsage_seq':
			sampler = UniformNeighborSampler(adj_info)
			layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
								SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

			model = SupervisedGraphsage(labels, self.seq.shape[0], placeholders, 
										 features,
										 context_features,
										 seq_info,
										 risk_type=self.FLAGS.risk_type,
										 layer_infos=layer_infos, 
										 aggregator_type="seq",
										 model_size=self.FLAGS.model_size,
										 sigmoid_loss = self.FLAGS.sigmoid,
										 identity_dim = self.FLAGS.identity_dim,
										 logging=True)

		elif self.FLAGS.model == 'graphsage_maxpool':
			sampler = UniformNeighborSampler(adj_info)
			layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
								SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

			model = SupervisedGraphsage(labels, self.seq.shape[0], placeholders, 
										features,
										seq_info,
										risk_type=self.FLAGS.risk_type,
										 layer_infos=layer_infos, 
										 aggregator_type="maxpool",
										 model_size=self.FLAGS.model_size,
										 sigmoid_loss = self.FLAGS.sigmoid,
										 identity_dim = self.FLAGS.identity_dim,
										 logging=True)

		elif self.FLAGS.model == 'graphsage_meanpool':
			sampler = UniformNeighborSampler(adj_info)
			layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
								SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

			model = SupervisedGraphsage(labels, self.seq.shape[0], placeholders, 
										features,
										seq_info,
										risk_type=self.FLAGS.risk_type,
										 layer_infos=layer_infos, 
										 aggregator_type="meanpool",
										 model_size=self.FLAGS.model_size,
										 sigmoid_loss = self.FLAGS.sigmoid,
										 identity_dim = self.FLAGS.identity_dim,
										 logging=True)

		else:
			raise Exception('Error: model name unrecognized.')

		sess = tf.compat.v1.Session()
		sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={adj_info_ph: self.adj,
																			seq_info_ph: self.seq})
		sess.run(model.features.assign(model.features_plhdr), feed_dict={model.features_plhdr: features})
		
		# Train model
		
		total_steps = 0
		avg_time = 0.0
		cost = 0.0

		train_adj_info = tf.compat.v1.assign(adj_info, self.adj)
		sess.graph.finalize()
		for epoch in range(self.FLAGS.epochs): 
			minibatch.shuffle() 

			iter = 0

			t = time.time()
			pbar = tqdm.tqdm(total=len(minibatch.train_links))
			sess.run(model.r_opp)
			while not minibatch.end():
				# Construct feed dictionary
				pbar.update(self.FLAGS.batch_size)
				feed_dict, labels = minibatch.next_minibatch_feed_dict()
				feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})

				# Training step
				if verbose and minibatch.end():
					outs = sess.run([ model.opt_op, model.loss], feed_dict=feed_dict)
					train_cost = outs[1] 
				else:
					outs = sess.run([ model.opt_op], feed_dict=feed_dict)


				iter += 1
				total_steps += 1

				if total_steps > self.FLAGS.max_total_steps:
					break
			avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
			
			pbar.close()

			if verbose:
				if self.FLAGS.validate_batch_size == -1:
					cost, val_f1_mac, val_lrap, _ , _, f1_pu,  duration = self.incremental_evaluate(sess, model, minibatch, self.FLAGS.batch_size)
				else:
					cost, val_f1_mac, val_lrap, f1_pu, duration = self.evaluate(sess, model, minibatch, self.FLAGS.validate_batch_size)
				
				print("Iter:", '%04d' % iter,
					  "train_loss=", "{:.5f}".format(train_cost),
					  "val_loss=", "{:.5f}".format(cost),
					  "val_f1_mac=", "{:.5f}".format(val_f1_mac),
					  "val_f1_pu=", "{:.5f}".format(f1_pu),
					  "val_lrap=", "{:.5f}".format(val_lrap),
					  "time=", "{:.5f}".format(avg_time)) 
			

			if total_steps > self.FLAGS.max_total_steps:
				break
		
		print("Optimization Finished!")
		

		print("Writing test set stats to file (don't peak!)")
		result_file = open("rez_output/{}_eval_results_{}.csv".format(self.FLAGS.data_folder, self.FLAGS.model), "w")
		for tests in ['full']:
			cost, test_f1_mac, test_lrap, test_f1_bi, test_auc, f1_pu, duration = self.incremental_evaluate(sess, model, minibatch, self.FLAGS.batch_size, test=True, test_set=tests)
			result_file.write("\n=============Result for {}============\n".format(tests.upper()))
			result_file.write("Full Test stats ({}):\nloss= {:.5f}\nf1_bi= {:.5f}\nf1_mac= {:.5f}\nf1_pu= {:.5f}\nauc= {:.5f}\nlrap= {:.5f}\ntime= {:.5f}".format(tests.upper(), 
				cost,test_f1_bi, test_f1_mac, f1_pu, test_auc, test_lrap, duration))

			
			print("\n=============Result for {}============\n".format(tests.upper()))
			print("Full Test stats ({}):\nloss= {:.5f}\nf1_bi= {:.5f}\nf1_mac= {:.5f}\nf1_pu= {:.5f}\nauc= {:.5f}\nlrap= {:.5f}\ntime= {:.5f}".format(tests.upper(), 
				cost,test_f1_bi, test_f1_mac, f1_pu, test_auc, test_lrap, duration))
			
			
			self.get_emb_and_plot(sess, model, minibatch, self.FLAGS.batch_size, test=True, test_set=tests)
		
			print(self.cid)
		result_file.close()
