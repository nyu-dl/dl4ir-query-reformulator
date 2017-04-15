# -*- coding: utf-8 -*-
'''
Custom theano class to query the search engine.
'''

import numpy as np
import nltk
import theano
from theano import gof
from theano import tensor
import time
import parameters as prm
import utils
from collections import OrderedDict
import average_precision
import supervised
import random


class Search(theano.Op):
    __props__ = ()

    def __init__(self,options):
        self.options = options
        self.options['reformulated_queries'] = {}


    def make_node(self, x1, x2, x3, x4, x5):
        # check that the theano version has support for __props__.
        # This next line looks like it has a typo,
        # but it's actually a way to detect the theano version
        # is sufficiently recent to support the use of __props__.
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x1 = tensor.as_tensor_variable(x1)
        x2 = tensor.as_tensor_variable(x2)
        x3 = tensor.as_tensor_variable(x3)
        x4 = tensor.as_tensor_variable(x4)
        x5 = tensor.as_tensor_variable(x5)
        out = [tensor.fmatrix().type(), tensor.itensor3().type(), tensor.ftensor3().type(), tensor.imatrix().type(), tensor.fmatrix().type()]

        if prm.supervised:
            out += [tensor.ftensor3().type()]

        return theano.Apply(self, [x1, x2, x3, x4, x5], out)


    def perform(self, node, inputs, output_storage):
        st = time.time()
        q_m = inputs[0]
        D_truth = inputs[1]
        n_iter = int(inputs[2])
        oracle_mode = int(inputs[3])
        is_train = int(inputs[4])

        #outputs
        metrics = np.zeros((len(q_m), len(prm.metrics_map)), np.float32)

        if is_train:
            max_feedback_docs = prm.max_feedback_docs_train
        else:
            max_feedback_docs = prm.max_feedback_docs

        if prm.cand_terms_source.lower() in ['syn', 'synemb']:
            cand_size = prm.syns_per_word
        elif prm.cand_terms_source.lower() == 'doc':
            cand_size = max_feedback_docs
        elif prm.cand_terms_source.lower() == 'all':
            cand_size = max_feedback_docs + prm.syns_per_word
        
        D_i = -2 * np.ones((len(q_m), cand_size, prm.max_words_input), np.int32)
        D_idf_m = np.ones((len(q_m), cand_size, prm.max_words_input), np.float32)
        D_gt_m = np.zeros((len(q_m), prm.max_candidates), np.float32)
        D_id = np.zeros((len(q_m), prm.max_candidates), np.int32)

        if prm.supervised:
            D_i_gt = np.zeros((len(q_m), max_feedback_docs, prm.max_words_input), np.float32)

        # no need to retrieve extra terms in the last iteration
        if n_iter == prm.n_iterations - 1:
            extra_terms = False
        else:
            extra_terms = True

        # allow the search engine to cache queries only in the first iteration.
        if n_iter == 0:
            save_cache = prm.use_cache
        else:
            save_cache = False
    
        max_cand = prm.max_candidates

        qs = []
        for i, q_lst in enumerate(self.options['current_queries']):
            #st1 = time.time()
            q = []
            for j, word in enumerate(q_lst):
                if q_m[i,j] == 1:
                    q.append(str(word))
            q = ' '.join(q)

            if len(q) == 0:
                q = 'dummy'
            #print 'query to lucene:', q
            qs.append(q)

        # only used to print the reformulated queries.
        self.options['reformulated_queries'][n_iter] = qs

        st1 = time.time()

        # always return one more candidate because one of them might be the input doc.
        candss = self.options['engine'].get_candidates(qs, max_cand, prm.max_feedback_docs, save_cache, extra_terms)
        st11 = time.time() - st1
        st22 = 0.
        st33 = 0.

        for i, cands in enumerate(candss):

            D_truth_dic = {}
            for d_truth in D_truth[i]:
                if d_truth > -1:
                    D_truth_dic[d_truth] = 0

            D_id[i,:len(cands.keys())] = cands.keys()

            j = 0
            m = 0
            all_txt = ''
            cand_ids = []

            if prm.supervised:
                wordss = []

            selected_docs = np.arange(prm.max_feedback_docs)

            if is_train and not prm.supervised:
                selected_docs = np.random.choice(selected_docs, size=prm.max_feedback_docs_train, replace=False)

            for k, (cand_id, (words_idx, words)) in enumerate(cands.items()):

                cand_ids.append(cand_id)

                if prm.cand_terms_source.lower() in ['doc', 'all']:
                    # no need to add candidate words in the last iteration.
                    if n_iter < prm.n_iterations - 1:
                        # only add docs selected by sampling (if training).
                        if k in selected_docs:
                            
                            words = words[:prm.max_terms_per_doc]
                            words_idx = words_idx[:prm.max_terms_per_doc]

                            D_i[i,m,:len(words_idx)] = words_idx

                            if prm.idf_threshold > 0.0:
                                for p, word in enumerate(words):
                                    if word.lower() in self.options['engine'].idf:
                                        if self.options['engine'].idf[word.lower()] <= prm.idf_threshold:
                                            D_idf_m[i,m,p] = 0

                            if prm.supervised:
                                wordss.append(words)

                            # append empty strings, so the list size becomes <dim>.
                            words = words + max(0, prm.max_words_input - len(words)) * ['']

                            # append new words to the list of current queries.
                            self.options['current_queries'][i] += words

                            m += 1

                if cand_id in D_truth_dic:
                    D_gt_m[i,j] = 1.
                
                j += 1

            if prm.cand_terms_source.lower() in ['syn', 'synemb', 'all']:
                # no need to add candidate words in the last iteration.
                if n_iter < prm.n_iterations - 1:

                    words = []
                    words_idx = []
                    for curr_word in self.options['current_queries'][i]:
                        curr_word = curr_word.decode('ascii', 'ignore')
                        if curr_word in self.options['syns']:
                            syns, syns_idx = self.options['syns'][curr_word]
                            words.extend(syns[:prm.syns_per_word])
                            words_idx.extend(syns_idx[:prm.syns_per_word])

                    words = words[:prm.syns_per_word * prm.max_words_input]
                    words_idx = words_idx[:prm.syns_per_word * prm.max_words_input]
                    temp = -2 * np.ones((prm.syns_per_word * prm.max_words_input))
                    temp[:len(words_idx)] = words_idx
                    temp = temp.reshape((prm.syns_per_word, prm.max_words_input))
                    D_i[i,-prm.syns_per_word:,:] = temp

                    # append empty strings, so the list size becomes <dim>.
                    words += max(0, prm.max_words_input - len(words)) * ['']

                    # append new words to the list of current queries.
                    self.options['current_queries'][i] += words


            cands_set = set(cands.keys())

            if qs[i].lower() in self.options['engine'].title_id_map:
                input_doc_id = self.options['engine'].title_id_map[qs[i].lower()]
                # Remove input doc from returned docs.
                # This operation does not raise an error if the element is not there.
                cands_set.discard(input_doc_id) 

            st2 = time.time()
            intersec = len(set(D_truth_dic.keys()) & cands_set)
            recall = intersec / max(1., float(len(D_truth_dic)))
            precision = intersec / max(1., float(prm.max_candidates))
            metrics[i,prm.metrics_map['RECALL']] = recall
            metrics[i,prm.metrics_map['PRECISION']] = precision
            metrics[i,prm.metrics_map['F1']] = 2 * recall * precision / max(0.01, recall + precision)
            avg_precision = average_precision.compute(D_truth_dic.keys(), cand_ids)
            metrics[i,prm.metrics_map['MAP']] = avg_precision
            metrics[i,prm.metrics_map['LOG-GMAP']] = np.log(avg_precision + 1e-5)

            if prm.supervised and n_iter == 0 and oracle_mode:
                d_i_gt = self.options['supervised'][self.options['current_queries_ids'][i]]
                if d_i_gt[0,0] == -2:
                    # only works with RECALL by now...
                    d_i_gt = supervised.run(qs[i], D_truth[i][0], wordss, metrics[i,prm.metrics_map[prm.reward.upper()]], self.options, D_gt=D_truth_dic.keys())

                    self.options['supervised'][self.options['current_queries_ids'][i]] = d_i_gt

                D_i_gt[i,:,:] = d_i_gt

        output_storage[0][0] = metrics
        output_storage[1][0] = D_i
        output_storage[2][0] = D_idf_m
        output_storage[3][0] = D_id
        output_storage[4][0] = D_gt_m

        if prm.supervised:
            output_storage[5][0] = D_i_gt

    def grad(self, inputs, output_grads):
        return [tensor.zeros_like(ii, dtype=theano.config.floatX) for ii in inputs]
        
