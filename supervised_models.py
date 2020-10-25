from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models_util as models
import layers as layers
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import numpy as np
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
tfd = tfp.distributions


tf.compat.v1.disable_eager_execution()
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, label, num_window,
            placeholders, features, seq,
            layer_infos, concat=True, risk_type="pn", aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - pairs: Node pairs.
            - labels: pair labels indicating existing connections.
            - num_window: Number of time window split.
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - seq: Numpy array with node id mapping for each time window
            - risk_type: Risk learning method {PN, UPU, NNPU}. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.label = label
        self.seq = seq
        self.prior = 0.5
        self.beta = 0
        self.gamma = 1
        if identity_dim > 0:
           self.embeds = tf.compat.v1.get_variable("node_embeddings", [features.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features_plhdr = tf.compat.v1.placeholder(dtype=tf.float32, shape=features.shape)
            self.features = tf.compat.v1.get_variable('feature', features.shape, trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)

        self.dtype = tf.float32
        self.concat = concat
        self.num_window = num_window
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.latent_dim = 64
        self.hidden_size = 128
        self.range = tf.Variable(tf.range(0, num_window, 1, dtype=tf.int32), trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.track_c  = tf.Variable([0,0], trainable=False)
        self.lambda_ = 0.0000001
        self._positive = 1
        self._negative = 0
        self._beta = 0
        self._gamma = 1
        self.risk_type = risk_type
        self.dim_mult = 2 if self.concat else 1



        self.v_weight = 1/label.shape[0]
        self.Nc = 2
        self.Nd = self.dim_mult*self.dims[-1]
        
        # Variational distribution variables for means
        self.locs = tf.Variable(tf.random.normal((self.Nc, self.Nd)))
        self.scales = tf.Variable(tf.pow(tf.random.gamma((self.Nc, self.Nd), 1, 1), -0.5))
        
        # Variational distribution variables for standard deviations
        self.alpha = tf.Variable(tf.random.uniform((self.Nc, self.Nd), 1., 2.))
        self.beta = tf.Variable(tf.random.uniform((self.Nc, self.Nd), 1., 2.))
        
        # Variational distribution variables for component weights
        self.counts = tf.Variable(2*tf.ones((self.Nc,)))

        # Prior distributions for the means
        self.mu_prior = tfd.Normal(tf.zeros((self.Nc, self.Nd)), tf.ones((self.Nc, self.Nd)))

        # Prior distributions for the standard deviations
        self.sigma_prior = tfd.Gamma(tf.ones((self.Nc, self.Nd)), tf.ones((self.Nc, self.Nd)))
        
        # Prior distributions for the component weights
        self.theta_prior = tfd.Dirichlet(2*tf.ones((self.Nc,)))

        self.ass_clust  = tf.zeros_like(0, dtype=tf.int64)
        self.r_opp = self.track_c.assign([0,0]) 

        self.build()

        dummy_emb = tf.zeros_like(tf.expand_dims(self.inputs1,-1), tf.float32)
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1,(self.dim_mult*self.dims[-1]))),
                        name='h_0' )
        losses = tf.zeros_like(0.0)   
        prior = tf.zeros_like(0.0) 
        self.pn_prior = tf.constant(self.estimate_pn_prior())    
        self.h_t, self.losses, self.prior = tf.scan(self.forward, self.range, initializer = [h_0, losses,prior], parallel_iterations=20, name='h_t_transposed', swap_memory=True )
        

        self.loss = tf.reduce_mean(self.losses) 
        grads_and_vars = self.optimizer.compute_gradients(loss=self.loss)

        tf.compat.v1.summary.scalar('loss', self.loss)

        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        # clipped_grads_and_vars = grads_and_vars
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
        self.preds = tf.nn.sigmoid(self.node_pred(self.h_t[-1,:,:]))

        flattened_emb = tf.reshape(self.h_t, [-1, self.dim_mult*self.dims[-1]])
        self.all_preds  = tf.reshape(tf.nn.sigmoid(self.node_pred(flattened_emb)), [num_window, -1, 1])

        
    def build(self):
        with tf.compat.v1.variable_scope("cost", reuse=tf.compat.v1.AUTO_REUSE):

            self.num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            
            self.recurrent = layers.Dense((self.dim_mult*self.dims[-1]*2), self.dim_mult*self.dims[-1]*2, 
                    dropout=self.placeholders['dropout'],
                    act=lambda x : x)
            self.current = layers.Dense((self.dim_mult*self.dims[-1]*2), (self.dim_mult*self.dims[-1]), 
                    dropout=self.placeholders['dropout'],
                    act=lambda x : x)

            self.node_pred = layers.Dense((self.dim_mult*self.dims[-1]), 1, 
                    dropout=self.placeholders['dropout'],
                    act=lambda x : x)

            self.prop_net = layers.Dense((self.dim_mult*self.dims[-1]), 1, 
                    dropout=self.placeholders['dropout'],
                    act=lambda x : x)

            self.aggregators = self.build_aggregators(self.num_samples, self.dims, concat=self.concat, model_size=self.model_size)
            
            self.recurrent_bais = tf.Variable(name='b' , initial_value = lambda: tf.initializers.ones()(self.dim_mult*self.dims[-1]))
            
    def estimate_pn_prior(self):
        pn_prior =  [(self.label <= i).sum()/self.label.shape[0] for i in range(self.num_window)]
        return pn_prior

    def forward(self, h, t ):


        def mode_n(ass_clust):
            unique, _, count = tf.unique_with_counts(ass_clust)
            return tf.scatter_nd(tf.reshape(unique,[-1,1]), count, shape=tf.constant([2]))

        htm1 = h[0]
        t_seq = self.seq[t]

        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, t)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos, t)

        samples1 = self.gather_list(t_seq, samples1)
        samples2 = self.gather_list(t_seq, samples2)

        outputs1 = self.aggregate(samples1, self.dims, self.num_samples,
                support_sizes1, self.aggregators, concat=self.concat, model_size=self.model_size)
        outputs2 = self.aggregate(samples2, self.dims, self.num_samples,
              support_sizes2, self.aggregators, concat=self.concat, model_size=self.model_size)
        
        self.outputs1 = tf.nn.l2_normalize(outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(outputs2, 1)

        x_t = tf.concat((self.outputs1,self.outputs2),-1)
        x_t = tf.nn.pool(tf.expand_dims(x_t, -1),[2],'MAX', padding='SAME', strides=[2])
        x_t = tf.squeeze(x_t, -1)


        zr_t = tf.keras.backend.hard_sigmoid(self.recurrent(tf.concat([x_t, htm1],-1)))
        z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)

        r_state = r_t * htm1
        h_proposal = tf.nn.tanh(self.current(tf.concat([x_t, r_state],-1)))

        h_t = tf.multiply(1 - z_t, htm1) + tf.multiply(z_t, h_proposal)  

        labels = tf.less_equal(self.placeholders['labels'], t)
        
        if self.risk_type != "pn":
            mu, sig, log_likelihoods, kl_sum, theta = self.GMM(h_t, False)
            x_post = tfd.Normal(loc=tf.reshape(tf.ones_like(labels, dtype=np.float32),[-1, 1, 1]) * mu,
                            scale=tf.reshape(tf.ones_like(labels,dtype=np.float32),[-1, 1, 1]) * tf.sqrt(sig))
            x_broadcasted = tf.tile(tf.reshape(h_t, [-1, 1, 256]), [1, 2, 1])
            # Sum over latent dimension.
            # ``log_liks`` ends up with shape (N, K).
            log_liks = x_post.log_prob(x_broadcasted) 
            log_liks = tf.reduce_sum(log_liks, 2)
            clusters = tf.argmax(log_liks, 1, output_type=tf.int32)

            has_pos = tf.reduce_any(labels)
            self.track_c.assign_add(tf.cond(has_pos, lambda:mode_n(clusters[labels]), lambda:tf.constant([0,0])))
            self.ass_clust = tf.math.argmax(self.track_c)
            prior = tf.gather(tf.reshape(tf.reduce_mean(theta,0),[-1]),self.ass_clust)
            labels = tf.reshape(tf.cast(labels, tf.float32),[-1,1])
            elbo_loss = self.v_weight* (tf.reduce_sum(kl_sum) - tf.reduce_sum(log_likelihoods))
        else:
            prior = self.pn_prior[t]
            labels = tf.reshape(tf.cast(labels, tf.float32),[-1,1])
            elbo_loss =  tf.constant(0.0)
        
        surrogate_loss = 'sigmoid'
        if self.risk_type == "pn":
            ex_loss = self.pn_loss(h_t, labels, prior, surrogate_loss, name='pn_loss')# + p_loss
        elif self.risk_type == "upu":
            ex_loss = self.upu_loss(h_t, labels, prior, surrogate_loss, name='pu_loss')
        else:
            ex_loss = self.nnpu_loss(h_t, labels, prior, surrogate_loss, name='pu_loss')

        loss = elbo_loss + ex_loss

        return [h_t, loss, prior]
    
    def nnpu_loss(self, h_t, labels, prior, surrogate_loss, name='nnpu_loss'):

        node_preds = self.node_pred(h_t)

        positive_risk, negative_risk = self._calculate_losses(node_preds,
                                                              labels,
                                                              prior,
                                                              surrogate_loss)

        is_ga = tf.less(negative_risk, -self._beta)
        loss_for_update = tf.cond(is_ga,
                                  lambda: positive_risk - self._beta,
                                  lambda: positive_risk + negative_risk)

        return loss_for_update

    def pn_loss(self, h_t, labels, prior, surrogate_loss, name='pn_loss'):
 
        node_preds = self.node_pred(h_t)
        loss = self._cal_pn_loss(node_preds, labels, prior, surrogate_loss)

        return loss


    def _calculate_losses(self, node_preds, labels, prop_score, surrogate_loss, logging=False):
        
        if not logging:
            loss_func = self._parse_loss_function(surrogate_loss)
        else:
            loss_func = self._parse_loss_function('zero-one')


        positive = tf.cast(tf.equal(labels, self._positive), tf.float32, name='positive_label')
        unlabeled = tf.cast(tf.equal(labels, self._negative), tf.float32, name='negative_label')
        num_positive = tf.maximum(1., tf.reduce_sum(positive), name='positive_number')
        num_unlabeled = tf.maximum(1., tf.reduce_sum(unlabeled), name='unlabeled_number')
        losses_positive = loss_func(node_preds, self._positive, 'positive_loss')
        losses_negative = loss_func(node_preds, -1, 'negative_loss')
        positive_risk = tf.reduce_mean(prop_score * positive / num_positive *
                                      losses_positive, name='positive_risk')
        negative_risk = tf.reduce_mean((unlabeled / num_unlabeled - prop_score *
                                       positive / num_positive) *
                                      losses_negative, name='negative_risk')



        return positive_risk, negative_risk


    def _parse_loss_function(self, surrogate_loss):
        assert surrogate_loss in ['sigmoid', 'logistics', 'zero-one']
        assert self._negative == 0 and self._positive == 1
        if surrogate_loss == 'sigmoid':
            return lambda network_out, y, name: \
                tf.nn.sigmoid(-network_out * y, name=name)
        elif surrogate_loss == 'logistics':
            return lambda network_out, y, name:\
                tf.nn.softplus(-network_out * y, name=name)
        elif surrogate_loss == 'zero-one':
            return lambda network_out, y, name:\
                (tf.cast(tf.greater(-network_out * y, 0.), tf.float32))
        else:
            raise NotImplementedError('Unknown loss function: {}'
                                      .format(surrogate_loss))

    def _cal_pn_loss(self, node_preds, labels, prop_score, surrogate_loss):
        positive = tf.cast(tf.equal(labels, self._positive), tf.float32,
                           name='positive_label')
        negative = tf.cast(tf.equal(labels, self._negative), tf.float32,
                           name='negative_label')


        num_positive = tf.maximum(1., tf.reduce_sum(positive), name='positive_number')
        num_negative = tf.maximum(1., tf.reduce_sum(negative), name='negative_number')

        loss_func = self._parse_loss_function(surrogate_loss)
        positive_losses = loss_func(node_preds, self._positive, 'positive_losses')
        negative_losses = loss_func(node_preds, -1, 'negative_losses')
        losses = tf.reduce_mean(prop_score * positive / num_positive *
                               positive_losses + (1 - prop_score) * negative /
                               num_negative * negative_losses, name='pu_risk')

        return losses

    def upu_loss(self, h_t, labels, prior, surrogate_loss, name='upu_loss'):

        node_preds = self.node_pred(h_t)

        positive_risk, negative_risk = self._calculate_losses(node_preds,
                                                              labels,
                                                              prior,
                                                              surrogate_loss)
        upu_loss = positive_risk + negative_risk
        return upu_loss


    def GMM(self, x, sampling=True):
        """Compute losses given a batch of data.
        
        Parameters
        ----------
        x : tf.Tensor
            A batch of data
        sampling : bool
            Whether to sample from the variational posterior
            distributions (if True, the default), or just use the
            mean of the variational distributions (if False).
            
        Returns
        -------
        log_likelihoods : tf.Tensor
            Log likelihood for each sample
        kl_sum : tf.Tensor
            Sum of the KL divergences between the variational
            distributions and their priors
        mu : tf.Tensor
            Mean of the GMM components
        sigma : tf.Tensor
            Standard diviation of the distributions
        theta : tf.Tensor
            Weight of the components in the mixture models
        """
        
        # The variational distributions
        mu = tfd.Normal(self.locs, self.scales)
        sigma = tfd.Gamma(self.alpha, self.beta)
        theta = tfd.Dirichlet(self.counts)
        
        # Sample from the variational distributions
        if sampling:
            Nb = tf.shape(x)[0] #number of samples in the batch
            mu_sample = mu.sample(Nb)
            sigma_sample = tf.pow(sigma.sample(Nb), -0.5)
            theta_sample = theta.sample(Nb)
        else:
            mu_sample = tf.reshape(mu.mean(), (1, self.Nc, self.Nd))
            sigma_sample = tf.pow(tf.reshape(sigma.mean(), (1, self.Nc, self.Nd)), -0.5)
            theta_sample = tf.reshape(theta.mean(), (1, self.Nc))
        
        # The mixture density
        density = tfd.Mixture(
            cat=tfd.Categorical(probs=theta_sample),
            components=[
                tfd.MultivariateNormalDiag(loc=mu_sample[:, i, :],
                                           scale_diag=sigma_sample[:, i, :])
                for i in range(self.Nc)])
                
        # Compute the mean log likelihood
        log_likelihoods = density.log_prob(x)
        
        # # Compute the KL divergence sum
        mu_div    = tf.reduce_sum(tfd.kl_divergence(mu,    self.mu_prior))
        sigma_div = tf.reduce_sum(tfd.kl_divergence(sigma, self.sigma_prior))
        theta_div = tf.reduce_sum(tfd.kl_divergence(theta, self.theta_prior))

        kl_sum = mu_div + sigma_div + theta_div
        
        return mu_sample, sigma_sample, log_likelihoods, kl_sum, theta_sample
