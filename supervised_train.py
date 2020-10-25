from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
from Hypothesis_train import Hypo_Gen
tf.compat.v1.disable_eager_execution()

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
tf.compat.v1.app.flags.DEFINE_boolean('log_device_placement', False,
							"""Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_maxpool', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string("key_type", "all_keys", "dataset category to use")
flags.DEFINE_string("data_folder", "cancer_data", "dataset to use")
flags.DEFINE_string("test_type","full", "method for evaluation [full,eval]")
flags.DEFINE_string("risk_type", "upu", "risk to use")

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0,'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 20, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0,'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
# flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 40, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

# os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'







def main(argv=None):
	acc, f1, rec, conf_mat = 0, 0, 0, np.array([[0,1],[0,0]])
	current_year = 1959
	verbose = True
	tf.compat.v1.reset_default_graph()
	hpg = Hypo_Gen(current_year, verbose)
	hpg.train_test(FLAGS, verbose)

if __name__ == '__main__':
	tf.compat.v1.app.run()
	main()






