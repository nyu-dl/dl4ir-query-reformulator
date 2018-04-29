# -*- coding: utf-8 -*-
'''
Build and train the query reformulator model
'''
import cPickle as pkl
import time
import numpy as np
import theano
import theano.tensor as tensor
import utils
import corpus_hdf5
import dataset_hdf5
import parameters as prm
import nltk
import random
import sys
import h5py
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
from sklearn.decomposition import PCA
from op_search import Search


reload(sys)
sys.setdefaultencoding('utf8')

# only print four decimals on float arrays.
np.set_printoptions(linewidth=150, formatter={'float': lambda x: "{0:0.4f}".format(x)})

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)



def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def get_minibatches_idx(n, minibatch_size, shuffle=False, max_samples=None):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    if max_samples:
        idx_list = idx_list[:max_samples]
        n = max_samples

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, is_train, trng):
    proj = tensor.switch(is_train,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=(1-prm.dropout), n=1,
                                        dtype=state_before.dtype)),
                         state_before * (1-prm.dropout))
    return proj


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk in pp:
            if params[kk].shape == pp[kk].shape:
                params[kk] = pp[kk]
            else:
                print 'The shape of layer', kk, params[kk].shape, 'is different from shape of the stored layer with the same name', pp[kk].shape, '.'
        else:
            print '%s is not in the archive' % kk

    return params


def load_wemb(params, vocab):
    wemb = pkl.load(open(prm.wordemb_path, 'rb'))
    dim_emb_orig = wemb.values()[0].shape[0]

    W = 0.01 * np.random.randn(prm.n_words, dim_emb_orig).astype(config.floatX)
    for word, pos in vocab.items():
        if word in wemb:
            W[pos,:] = wemb[word]
    
    if prm.dim_emb < dim_emb_orig:
        pca =PCA(n_components=prm.dim_emb, copy=False, whiten=True)
        W = pca.fit_transform(W)

    params['W'] = W

    return params


def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def matrix(dim):
    return np.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)


def softmax_mask(x, mask=None):
    m = tensor.max(x, axis=-1, keepdims=True)
    if mask:
        e_x = tensor.exp(x - m) * mask
    else:
        e_x = tensor.exp(x - m)
    return e_x / tensor.maximum(e_x.sum(axis=-1, keepdims=True), 1e-8) #this small constant avoids possible division by zero created by the mask


def lstm_layer(x, h_, c_):
       
    i = tensor.nnet.sigmoid(_slice(x, 0, prm.dim_proj))
    f = tensor.nnet.sigmoid(_slice(x, 1, prm.dim_proj))
    o = tensor.nnet.sigmoid(_slice(x, 2, prm.dim_proj))
    c = tensor.tanh(_slice(x, 3, prm.dim_proj))

    c = f * c_ + i * c
    h = o * tensor.tanh(c)
    return h, c


def init_params(options):
    params = OrderedDict()
    exclude_params = {}

    params['W'] = 0.01 * np.random.randn(prm.n_words, prm.dim_emb).astype(config.floatX) # vocab to word embeddings
    params['UNK'] = 0.01 * np.random.randn(1, prm.dim_emb).astype(config.floatX) # vector for unknown words.

    n_features = [prm.dim_emb,] + prm.filters_query
    for i in range(len(prm.filters_query)):
        params['Ww_att_q'+str(i)] = 0.01 * np.random.randn(n_features[i+1], n_features[i], 1, prm.window_query[i]).astype(config.floatX)
        params['bw_att_q'+str(i)] = np.zeros((n_features[i+1],)).astype(config.floatX) # bias score

    params['Aq'] = 0.01 * np.random.randn(n_features[-1], prm.dim_proj).astype(config.floatX) # score

    n_hidden_actor = [prm.dim_proj] + prm.n_hidden_actor + [2]
    for i in range(len(n_hidden_actor)-1):
        params['V'+str(i)] = 0.01 * np.random.randn(n_hidden_actor[i], n_hidden_actor[i+1]).astype(config.floatX) # score
        params['bV'+str(i)] = np.zeros((n_hidden_actor[i+1],)).astype(config.floatX) # bias score

    # set initial bias towards not selecting words.
    params['bV'+str(i)] = np.array([10., 0.]).astype(config.floatX) # bias score
    
    n_hidden_critic = [prm.dim_proj] + prm.n_hidden_critic + [1]
    for i in range(len(n_hidden_critic)-1):
        params['C'+str(i)] = 0.01 * np.random.randn(n_hidden_critic[i], n_hidden_critic[i+1]).astype(config.floatX) # score
        params['bC'+str(i)] = np.zeros((n_hidden_critic[i+1],)).astype(config.floatX) # bias score

    n_features = [prm.dim_emb,] + prm.filters_cand
    for i in range(len(prm.filters_cand)):
        params['Ww_att_c_0_'+str(i)] = 0.01 * np.random.randn(n_features[i+1], n_features[i], 1, prm.window_cand[i]).astype(config.floatX)
        params['bw_att_c_0_'+str(i)] = np.zeros((n_features[i+1],)).astype(config.floatX) # bias score

    params['Ad'] = 0.01 * np.random.randn(n_features[-1], prm.dim_proj).astype(config.floatX) # score
    params['bAd'] = np.zeros((prm.dim_proj,)).astype(config.floatX) # bias score

    if prm.fixed_wemb:
        exclude_params['W'] = True
    
    return params, exclude_params


def conv_query(q_a, tparams):

    q_aw = q_a.dimshuffle(0, 2, 'x', 1)
    for j in range(len(prm.filters_query)):
        q_aw = tensor.nnet.conv2d(q_aw,
                                    tparams['Ww_att_q'+str(j)],
                                    border_mode=(0, prm.window_query[j]//2))
        q_aw += tparams['bw_att_q'+str(j)][None,:,None,None]
        q_aw = tensor.maximum(q_aw, 0.)
    q_aw = q_aw[:, :, 0, :].dimshuffle(0, 2, 1)
    q_a = q_aw.reshape((q_a.shape[0], q_a.shape[1], -1))

    return q_a


def conv_cand(D_a, tparams, n_iter):

    D_aw = D_a.reshape((-1, D_a.shape[2], D_a.shape[3]))
    D_aw = D_aw.dimshuffle(0, 2, 'x', 1)
    for j in range(len(prm.filters_cand)):
        D_aw = tensor.nnet.conv2d(D_aw,
                                    tparams['Ww_att_c_' + str(n_iter) + '_' + str(j)],
                                    border_mode=(0, prm.window_cand[j]//2))
        D_aw += tparams['bw_att_c_' + str(n_iter) + '_' + str(j)][None,:,None,None]
        D_aw = tensor.maximum(D_aw, 0.)
    D_aw = D_aw[:, :, 0, :].dimshuffle(0, 2, 1)
    D_a = D_aw.reshape((D_a.shape[0], D_a.shape[1], D_a.shape[2], -1))

    return D_a


def f(q_i, D_gt_id, tparams, is_train, trng, options):

    # Use search engine again to compute the reward/metrics given a query.
    search = Search(options)

    # append the unknown vector for words whose index = -1.
    W_ = tensor.concatenate([tparams['W'], tparams['UNK']], axis=0)

    q_m = (q_i > -2).astype('float32')

    #get embeddings for the queries
    q_a = W_[q_i.flatten()].reshape((q_i.shape[0], q_i.shape[1], prm.dim_emb)) * q_m[:,:,None]

    if len(prm.filters_query) > 0:
        q_aa = conv_query(q_a, tparams)
    else:
        q_aa = q_a

    out = []
    for n_iter in range(prm.n_iterations):

        if n_iter == 0 and prm.q_0_fixed_until >= prm.n_iterations:
            prob = tensor.zeros((q_a.shape[0], prm.max_words_input, 2))
            bl = tensor.zeros((q_a.shape[0],))
            D_m_r = tensor.zeros((q_a.shape[0], prm.max_words_input))
        else:
            if n_iter > 0:
                D_m_ = (D_i_ > -2).astype('float32')
                D_a_ = W_[D_i_.flatten()].reshape((D_i_.shape[0], D_i_.shape[1], D_i_.shape[2], prm.dim_emb)) * D_m_[:,:,:,None]
            else:
                D_a_ = 1. * q_a[:,None,:,:]
                D_m_ = 1. * q_m[:,None,:]


            if len(prm.filters_cand) > 0:
                D_aa_ = conv_cand(D_a_, tparams, 0)
            else:
                D_aa_ = D_a_

            D_aa_ = tensor.dot(D_aa_, tparams['Ad']) + tparams['bAd']

            if n_iter > 0:
                if prm.q_0_fixed_until < 2:
                    D_a = tensor.concatenate([D_a, D_a_], axis=1)
                    D_aa = tensor.concatenate([D_aa, D_aa_], axis=1)
                    D_m = tensor.concatenate([D_m, D_m_], axis=1)
                else:
                    D_a = D_a_
                    D_aa = D_aa_
                    D_m = D_m_
            else:
                D_a = D_a_
                D_aa = D_aa_
                D_m = D_m_

            D_a_r = D_a.reshape((D_a.shape[0], -1, D_a.shape[3]))
            D_aa_r = D_aa.reshape((D_aa.shape[0], -1, D_aa.shape[3]))

            D_m_r = D_m.reshape((D_m.shape[0],-1))

       
            q_aa_avg = q_aa.sum(1) / tensor.maximum(1., q_m.sum(1, keepdims=True))
            q_aa_att = q_aa_avg[:,None,:]
            q_aa_att = tensor.dot(q_aa_att, tparams['Aq'])

            z = D_aa_r + q_aa_att

            # estimate reward based on the query.
            bl = theano.gradient.grad_scale(z, 0.1)
            D_m_r_c = theano.gradient.disconnected_grad(D_m_r)
            bl = bl.sum(1) / tensor.maximum(1., D_m_r_c.sum(1))[:,None]
            for i in range(len(prm.n_hidden_critic)+1):
                if prm.dropout > 0:
                    bl = dropout_layer(bl, is_train, trng)
                bl = tensor.maximum(0., bl)
                bl = tensor.dot(bl, tparams['C'+str(i)]) + tparams['bC'+str(i)]

            bl = tensor.tanh(bl)
            bl = bl.flatten()
    

            for i in range(len(prm.n_hidden_actor)+1):
                if prm.dropout > 0:
                    z = dropout_layer(z, is_train, trng)
                z = tensor.maximum(0., z)
                z = tensor.dot(z, tparams['V'+str(i)]) + tparams['bV'+str(i)]

            prob = softmax_mask(z) * D_m_r[:,:,None]

            # if training, sample. Otherwise, pick maximum probability.
            s = trng.multinomial(n=1, pvals=prob.reshape((-1, 2)), dtype=prob.dtype)
            s = s.reshape((prob.shape[0],prob.shape[1],prob.shape[2]))

            #if frozen is enabled and this iteration is within its limit, pick maximum probability.
            if prm.frozen_until > 0:
                if n_iter < prm.frozen_until:
                    s = prob

            res = tensor.eq(is_train,1.) * s + tensor.eq(is_train,0.) * prob

            # final answer & valid words
            ans = res.argmax(2) * D_m_r

        if n_iter < prm.q_0_fixed_until:
            ones = tensor.ones((q_a.shape[0], prm.max_words_input))
            if n_iter > 0:
                # select everything from the original query in the first iteration.
                ans = tensor.concatenate([ones, ans], axis=1)
            else:
                ans = ones

        metrics, D_i_, D_id_, D_gt_m_ = search(ans, D_gt_id, n_iter, is_train)

        out.append([prob, ans, metrics, bl, D_m_r, D_id_])

    return out


def sgd(lr, tparams, grads, iin, out, updates):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(iin, out, updates=gsup + updates,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

      
def adadelta(lr, tparams, grads, iin, out, updates):
    """
    An adaptive learning rate optimizer

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(iin, out, updates=zgup + rg2up + updates,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

      
def rmsprop(lr, tparams, grads, iin, out, updates):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(iin, out,
                                    updates=zgup + rgup + rg2up + updates,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def adam(lr0, tparams, grads, iin, out, updates):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    
    f_grad_shared = theano.function(iin, out, updates=gsup+updates, \
                                    on_unused_input='ignore', allow_input_downcast=True)

    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr0], lr_t, updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)
    off = 1e-8  # small constant to avoid log 0 = -inf
    consider_constant = []

    is_train = tensor.fscalar('is_train') # if = 1, training time.
    q_i = tensor.imatrix('q_i') # input query
    D_gt_id = tensor.imatrix('D_gt_id')

    out = f(q_i, D_gt_id, tparams, is_train, trng, options)
    
    cost = 0.
    out_p = []
    out_s = []
    reward_last = 0
    for i, (prob, ans, metrics, bl, D_m_r, D_id) in enumerate(out):
        learn = True

        # if frozen until is enabled.
        if prm.frozen_until > 0:
            # do not learn if this iteration is less than frozen_until.
            if i < prm.frozen_until:
                learn = False
        
        reward = metrics[:,prm.metrics_map[prm.reward.upper()]]

        r = reward - reward_last - bl
        # cost for the baseline
        cost_bl = (r ** 2).sum()

        if learn:
            if i < prm.q_0_fixed_until:
                cap = prm.max_words_input
            else:
                cap = 0

            r_ = theano.gradient.disconnected_grad(r)
            cost_i = r_[:,None] * (-tensor.log(prob[:,:,1] + off)) * ans[:,cap:]
            cost += cost_i.sum()
            cost += cost_bl
            
            # entropy regularization
            if prm.erate > 0.:
                cost_ent = prm.erate * ((D_m_r[:,:,None] * prob * tensor.log(prob + off)).sum(axis=(1,2))).sum()
                cost += cost_ent
            else:
                cost_ent = 0. * cost

        reward_last = reward

        out_p.extend([ans, metrics, D_id])
        out_s.extend([prob, ans, metrics, bl, cost_bl, D_id])

    if prm.l2reg > 0.:
        cost_l2reg = 0.
        for name, w in tparams.items():
            #do not include bias.
            if (not name.lower().startswith('b')) and (name.lower() != 'w'):
                cost_l2reg += prm.l2reg * (w**2).sum()
        cost += cost_l2reg

    f_pred = theano.function([q_i, D_gt_id, is_train], out_p, updates=[], name='f_pred', on_unused_input='ignore')

    iin = [q_i, D_gt_id, is_train]
    out = [cost, cost_ent] + out_s

    updates = []

    return iin, out, updates, f_pred, consider_constant


def lst2matrix(lst):
    maxdim = len(max(lst,key=len))
    out = -np.ones((len(lst), maxdim), dtype=np.int32)
    for i,item in enumerate(lst):
        out[i,:min(len(item),maxdim)] = item[:maxdim]
    return out


def get_samples(input_queries, target_docs, index, options):
    qi = [utils.clean(input_queries[t].lower()) for t in index]
    D_gt_title = [target_docs[t] for t in index]

    D_gt_id_lst = []
    for j, t in enumerate(index):
        D_gt_id_lst.append([])
        for title in D_gt_title[j]:
            if title in options['engine'].title_id_map:
                D_gt_id_lst[-1].append(options['engine'].title_id_map[title])
            else:
                print 'ground-truth doc not in index:', title

    D_gt_id = lst2matrix(D_gt_id_lst)
    
    qi_i, qi_lst_ = utils.text2idx2(qi, options['vocab'], prm.max_words_input)
    
    qi_lst = []
    for qii_lst in qi_lst_:
        # append empty strings, so the list size becomes <dim>.
        qi_lst.append(qii_lst + max(0, prm.max_words_input - len(qii_lst)) * [''])

    return qi, qi_i, qi_lst, D_gt_id, D_gt_title


def pred_error(f_pred, input_queries, target_docs, options, iterator):
    """
    Evaluate model on the metrics.
    f_pred: Theano function computing the prediction
    """

    n = 0.
    metrics = np.zeros((prm.n_iterations,len(prm.metrics_map)), dtype=np.float32)

    i = 0.
    for _, index in iterator:

        qi, qi_i, qi_lst, D_gt_id, D_gt_url = get_samples(input_queries, target_docs, index, options)

        # share the current queries with the search engine.
        options['current_queries'] = qi_lst

        is_train = 0.

        out = f_pred(qi_i, D_gt_id, is_train)
        
        if i % prm.dispFreq == 0:
            print '=================================================================='
            print
            print 'Input Query:         ', qi[0].replace('\n','\\n')
            print
            print 'Target Docs:          ', str(D_gt_url[0])
            print


        for j in range(prm.n_iterations):
            ans = out.pop(0)
            metrics_i = out.pop(0)
            D_id = out.pop(0)
            metrics[j] += metrics_i.sum(0)

            if i % prm.dispFreq == 0:
                print 
                print 'Iteration', j
                print
                print 'Retrieved Docs:    ', str([options['engine'].id_title_map[d_id] for d_id in D_id[0]])
                print
                print 'Reformulated Query:', options['reformulated_queries'][j][0]
                print 
                print 'Query ANS:         ',
                for kk, word in enumerate(options['current_queries'][0][:ans.shape[1]]):                         
                    if word not in options['vocab'] and word != '':
                        word += '<unk>'
                    if ans[0,kk] == 1:
                        word = word.upper()
                    print str(word), 
                print

        if i % prm.dispFreq == 0:
            print '=================================================================='

        n += len(index)
        i += 1.

    metrics /= n

    return metrics


def train():

    if prm.optimizer.lower() == 'adam':
        optimizer=adam
    elif prm.optimizer.lower() == 'sgd':
        optimizer=sgd
    elif prm.optimizer.lower() == 'rmsprop':
        optimizer=rmsprop
    elif prm.optimizer.lower() == 'adadelta':
        optimizer=adadelta

    options = locals().copy()

    print 'parameters:', str(options)
    prm_k = vars(prm).keys()
    prm_d = vars(prm)
    prm_k.sort()
    for x in prm_k:
        if not x.startswith('__'):
            print x,'=', prm_d[x]

    print 'loading Vocabulary...'
    vocab = utils.load_vocab(prm.vocab_path, prm.n_words)
    options['vocab'] = vocab

    options['vocabinv'] = {}
    for k,v in vocab.items():
        options['vocabinv'][v] = k

    print 'Loading Environment...'
    if prm.engine.lower() == 'lucene':
        import lucene_search
        options['engine'] = lucene_search.LuceneSearch()
    elif prm.engine.lower() == 'elastic':
        import elastic_search
        options['engine'] = elastic_search.ElasticSearch()

    print 'Loading Dataset...'
    dh5 = dataset_hdf5.DatasetHDF5(prm.dataset_path)
    qi_train, qi_valid, qi_test = dh5.get_queries()
    dt_train, dt_valid, dt_test = dh5.get_doc_ids()
    
    if prm.train_size == -1:
        train_size = len(qi_train)
    else:
        train_size = min(prm.train_size, len(qi_train))

    if prm.valid_size == -1:
        valid_size = len(qi_valid)
    else:
        valid_size = min(prm.valid_size, len(qi_valid))

    if prm.test_size == -1:
        test_size = len(qi_test)
    else:
        test_size = min(prm.test_size, len(qi_test))

    print '%d train examples' % len(qi_train)
    print '%d valid examples' % len(qi_valid)
    print '%d test examples' % len(qi_test)

    # This create the initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params, exclude_params = init_params(options)

    if prm.wordemb_path:
        print 'loading pre-trained word embeddings'
        params = load_wemb(params, vocab)
        options['W'] = params['W']

    if prm.reload_model:
        load_params(prm.reload_model, params)

    print 'Building model'
    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    iin, out, updates, f_pred, consider_constant \
            = build_model(tparams, options)

    #get only parameters that are not in the exclude_params list
    tparams_ = OrderedDict([(kk, vv) for kk, vv in tparams.iteritems() if kk not in exclude_params])

    grads = tensor.grad(out[0], wrt=itemlist(tparams_), consider_constant=consider_constant)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams_, grads, iin, out, updates)

    history_errs = []
    best_p = None

    if prm.validFreq == -1:
        validFreq = len(qi_train) / prm.batch_size_train
    else:
        validFreq = prm.validFreq

    if prm.saveFreq == -1:
        saveFreq = len(qi_train) / prm.batch_size_train
    else:
        saveFreq = prm.saveFreq

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    print 'Optimization'
    
    try:
        for eidx in xrange(prm.max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(qi_train), prm.batch_size_train, shuffle=True)

            for _, train_index in kf:
                st = time.time()

                uidx += 1
                qi, qi_i, qi_lst, D_gt_id, D_gt_url = get_samples(qi_train, dt_train, train_index, options)

                # share the current queries with the search engine.
                options['current_queries'] = qi_lst

                n_samples += len(qi)

                is_train = 1.

                out = f_grad_shared(qi_i, D_gt_id, is_train)

                cost = out.pop(0)
                cost_ent = out.pop(0)

                lr_t = f_update(prm.lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.
    
                if np.mod(uidx, prm.dispFreq) == 0:

                    print '\n================================================================================'
                    print 'Epoch', eidx, 'Update', uidx, 'Cost', cost, 'LR_t', lr_t
                    print 'Time Minibatch Update: ' + str(time.time() - st)
                    print 'Input Query:       ', qi[0].replace('\n','\\n')
                    print
                    print 'Target Docs:       ', str(D_gt_url[0])
                    print
                    print 'Input Query Vocab: ', utils.idx2text(qi_i[0], options['vocabinv'])
                    for ii in range(prm.n_iterations):
                        prob = out.pop(0)
                        ans = out.pop(0)
                        metrics = out.pop(0)
                        bl = out.pop(0)
                        cost_bl = out.pop(0)
                        D_id = out.pop(0)
                        print 
                        print 'Iteration', ii
                        print 'Baseline Value', bl.mean(), 'Cost', cost_bl
                        print '  '.join(prm.metrics_map.keys())
                        print metrics.mean(0)
                        print
                        print 'Retrieved Docs:    ', str([options['engine'].id_title_map[d_id] for d_id in D_id[0]])
                        print
                        print 'Reformulated Query:', options['reformulated_queries'][ii][0]
                        print
                        print 'Query ANS:         ',
                        for kk, word in enumerate(options['current_queries'][0][:ans.shape[1]]):                         
                            if word not in options['vocab'] and word != '':
                                word += '<unk>'
                            if ans[0,kk] == 1:
                                word = word.upper()
                            print str(word), 
                        print
                        print
                        print 'prob[:,:,0].max(1).mean(), prob[:,:,0].mean(), prob[:,:,0].min(1).mean()', prob[:,:,0].max(1).mean(), prob[:,:,0].mean(), prob[:,:,0].min(1).mean()
                        print 'prob[:,:,1].max(1).mean(), prob[:,:,1].mean(), prob[:,:,1].min(1).mean()', prob[:,:,1].max(1).mean(), prob[:,:,1].mean(), prob[:,:,1].min(1).mean()
                    print '==================================================================================\n'


                if np.mod(uidx, validFreq) == 0 or uidx == 1:
             
                    kf_train = get_minibatches_idx(len(qi_train), prm.batch_size_pred, shuffle=True, max_samples=train_size)
                    kf_valid = get_minibatches_idx(len(qi_valid), prm.batch_size_pred, shuffle=True, max_samples=valid_size)
                    kf_test = get_minibatches_idx(len(qi_test), prm.batch_size_pred, shuffle=True, max_samples=test_size)

                    print '\nEvaluating - Training Set'
                    train_metrics = pred_error(f_pred, qi_train, dt_train, options, kf_train)

                    print '\nEvaluating - Validation Set'
                    valid_metrics = pred_error(f_pred, qi_valid, dt_valid, options, kf_valid)

                    print '\nEvaluating - Test Set'
                    test_metrics = pred_error(f_pred, qi_test, dt_test, options, kf_test)


                    his = [train_metrics, valid_metrics, test_metrics]
                    history_errs.append(his)
                    metric_idx = prm.metrics_map[prm.reward.upper()]
                    if (uidx == 0 or
                        valid_metrics[-1, metric_idx] >= np.array(history_errs)[:,1,-1,metric_idx].max()):

                        best_p = unzip(tparams)
                        bad_counter = 0


                    print '====================================================================================================='
                    print '  '.join(prm.metrics_map.keys())
                    print
                    print 'Train:'
                    print train_metrics
                    print
                    print 'Valid:'
                    print valid_metrics
                    print
                    print 'Test:'
                    print test_metrics
                    print
                    print '====================================================================================================='
                    if (len(history_errs) > prm.patience and
                        valid_metrics[-1, metric_idx] <= np.array(history_errs)[:-prm.patience,
                                                               1,-1,metric_idx].max()):
                        bad_counter += 1
                        if bad_counter > prm.patience:
                            print 'Early Stop!'
                            estop = True
                            break

                if prm.saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(prm.saveto, history_errs=history_errs, **params)

                    print 'Done'

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    return


if __name__ == '__main__':
    # See parameters.py for all possible parameter and their definitions.
    train()
