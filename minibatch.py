from __future__ import division
from __future__ import print_function

import numpy as np
import random
random.seed(123)
import pdb

np.random.seed(123)

class MinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    rain_edges -- link pairs (edges)
    """
    def __init__(self, edges, labels, masks, 
            placeholders, num_classes, max_id, batch_size=100, max_degree=128,
            **kwargs):

        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.num_classes = num_classes
        self.max_id = max_id


        train_mask, self.test_mask, self.unk_test_mask, self.corona = masks
        self.edges = edges
        self.labels = labels
        self.lastlabel = np.sort(np.unique(labels))[-2]
        self.full_train_labels = self.labels[train_mask]
        self.val_labels = self.full_train_labels[:20]

        self.full_train_links = self.edges[train_mask, :]
        self.val_links =  self.full_train_links[:20]
        self.val_set_size = len(self.val_links)
        
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_links)

    def batch_feed_dict(self, batch_links, batch_labels, is_train=False):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_links:
            batch1.append(node1)
            batch2.append(node2)

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_links)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['labels']: batch_labels})
        feed_dict.update({self.placeholders['is_train']: is_train})
        
        return feed_dict, batch_labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_links))
        batch_links = self.train_links[start_idx : end_idx]
        batch_labels = self.train_labels[start_idx : end_idx]

        return self.batch_feed_dict(batch_links, batch_labels, True)

    def num_training_batches(self):
        return len(self.train_links) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        if size is None:
            return self.batch_feed_dict(self.val_links, self.val_labels)
        else:
            val_size = len(self.val_labels)
            val_edges = self.val_links[:min(size, val_size)]
            val_labels = self.val_labels[:min(size, val_size)]
            return self.batch_feed_dict(val_edges, val_labels)

    def set_test_data(self, test_set='full', corona = False):
        self.edge_list = np.concatenate((self.edges[list(self.test_mask[test_set])] , self.edges[list(self.unk_test_mask[test_set])]), 0)
        self.edge_lables = np.concatenate((self.labels[list(self.test_mask[test_set])] , self.labels[list(self.unk_test_mask[test_set])]), 0)

        if corona:
            corona = [i for i,e in enumerate(list(self.test_mask[test_set])) if e in self.corona]
            return corona
        else:
            return None

    def set_eval_data(self, test_set='full', corona = False):
        corona = [i for i,e in enumerate(list(self.test_mask[test_set])) if e in self.corona]
        self.edge_list = self.edges[list(self.corona)] 
        self.edge_lables = self.labels[list(self.corona)]
        
        return None

    def incremental_val_feed_dict(self, size, iter_num, test=False):
        if not test and iter_num == 0:
            self.edge_list= self.val_links
            self.edge_lables = self.val_labels
        
        edges = self.edge_list[iter_num*size:min((iter_num+1)*size, 
            len(self.edge_list))]

        labels = self.edge_lables[iter_num*size:min((iter_num+1)*size, 
            len(self.edge_lables))]


        ret_val = self.batch_feed_dict(edges, labels)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(self.edge_lables), edges, labels

    

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        N = self.full_train_links.shape[0]
        ids = np.arange(N)#, int(N*self.pos_sample_rate), False )
        ids = np.random.permutation(ids)
        self.train_links = self.full_train_links[ids]
        self.train_labels = self.full_train_labels[ids]
        self.batch_num = 0
