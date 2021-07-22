# Temporal Positive-unlabeled Learning for Biomedical Hypothesis Generation via Risk Estimation

### Overview

This directory contains code necessary to run the TRP algorithm.
TRP is a positive-unlabeled (PU) learning algorithm for semi-supervised relationship prediction between node-pairs, which aims at integrating information from temporal structure of the network into the learning process of node pair embeddings in attributed graphs. The learned node
embeddings are used to conduct both transductive and inductive node pair relationship prediction.

See our [paper](https://arxiv.org/pdf/2010.01916.pdf) for details on the algorithm.

The datasets  used in our experiments can be downloaded from [Link](https://drive.google.com/file/d/15ODnZcpUZCSxEuyS7TbCKh9tIHjoRIWR/view?usp=sharing).

If you make use of this code or the MLGW algorithm in your work, please cite the following paper:

	@inproceedings{akujuobi2020temporal,
	     author = {Akujuobi, Uchenna and Chen, Jun and Elhoseiny, Mohamed and Spranger, Michael and Zhang, Xiangliang},
	     title = {Temporal Positive-unlabeled Learning for Biomedical Hypothesis Generation via Risk Estimation},
	     booktitle = {NeurIPS 2020},
	     year = {2020}
	  }

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn are required. You can install all the required packages using the following command:

	$ pip install -r requirements.txt


### Running the code

Use `python supervised_train.py` to run using default settings. The parameters can be changed by passing during the command call (e.g., `python supervised_train.py`). Use `python supervised_train.py --help` to display the parameters.

#### Input format
At minimums, the code requires that a `--data_folder` option is specified which specifies the following data files:

* <data_folder>/graph/tgraph_xxxx.edgelist -- An edgelist file describing the input graph. This is a three columned file with format `<source node>\t<target node>\t<co-occurence count>`.
* <data_folder>/emb/matrices/lsi/*.npz -- A numpy-stored array of node and context features ordered according to the node index in the edgelist file.
* <data_folder>/id_maps/*  -- node id ordered according to the node appearance in the corresponding edgelist files.
* <data_folder>/text_content/* -- node contexts indicies in the corresponding context feature numpy-stored arrays.

To run the model on a new dataset, you need to make data files in the format described above.

#### Using the outputs
To plot the node pair embeddings get print the specified node predictions, please use the `--analyze_pred` option. The ouput will be stored in an "output" folder.


#### Dataset Download

The COVID-19 dataset used in this work was obtained from https://www.semanticscholar.org/cord19/download while the Immunotherapy and virology papers were extracted from the full PubMed dump https://www.nlm.nih.gov/databases/download/pubmed_medline.html 
The biomedical term dataset can be downloaded from ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/  

The file “keylist” is extracted from the biomedical term dataset. The file “m_keylist” was extracted from http://blender.cs.illinois.edu/covid19/ 

The file “keylist” have 4 columns (From left to right) : term_type, term_id, term_name,  term_description

The file “m_keylist” have 5 columns (From left to right) : term_type, term_id, term_source, term_name, term_description

term_type signifies the term category, term_id is the ID assigned to the term, term_name is the term, and the term_description is the descritption of the term. The m_keylist includes an additional column “term_source” signifies the data source.


