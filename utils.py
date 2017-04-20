'''
Miscellaneous functions.
'''

import numpy as np
import cPickle as pkl
from nltk.tokenize import wordpunct_tokenize
import parameters as prm
from random import randint
import math
import re


def clean(txt):
    '''
    #remove most of Wikipedia and AQUAINT markups, such as '[[', and ']]'.
    '''
    txt = re.sub(r'\|.*?\]\]', '', txt) # remove link anchor

    txt = txt.replace('&amp;', ' ').replace('&lt;',' ').replace('&gt;',' ').replace('&quot;', ' ').replace('\'', ' ').replace('(', ' ').replace(')', ' ').replace('.', ' ').replace('"',' ').replace(',',' ').replace(';',' ').replace(':',' ').replace('<93>', ' ').replace('<98>', ' ').replace('<99>',' ').replace('<9f>',' ').replace('<80>',' ').replace('<82>',' ').replace('<83>', ' ').replace('<84>', ' ').replace('<85>', ' ').replace('<89>', ' ').replace('=', ' ').replace('*', ' ').replace('\n', ' ').replace('!', ' ').replace('-',' ').replace('[[', ' ').replace(']]', ' ')

    return txt


def BOW(words, vocab):
    '''
    Convert a list of words to the BoW representation.
    '''
    bow = {} # BoW densely represented as <vocab word idx: quantity>
    for word in words:
        if word in vocab:
            if vocab[word] not in bow:
                bow[vocab[word]] = 0.
            bow[vocab[word]] += 1.

    bow_v = np.asarray(bow.values())
    sumw = float(bow_v.sum())
    if sumw == 0.:
        sumw = 1.
    bow_v /= sumw

    return [bow.keys(), bow_v]


def BOW2(texts, vocab, dim):
    '''
    Convert a list of texts to the BoW dense representation.
    '''
    out = np.zeros((len(texts), dim), dtype=np.int32)
    mask = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        bow = BOW(wordpunct_tokenize(text), vocab)
        out[i,:len(bow[0])] = bow[0]
        mask[i,:len(bow[1])] = bow[1]

    return out, mask


def Word2Vec_encode(texts, wemb):
    
    out = np.zeros((len(texts), prm.dim_emb), dtype=np.float32)
    for i, text in enumerate(texts):
        words = wordpunct_tokenize(text)
        n = 0.
        for word in words:
            if word in wemb:
                out[i,:] += wemb[word]
                n += 1.
        out[i,:] /= max(1.,n)

    return out


def text2idx(texts, vocab, dim, use_mask=False):
    '''
    Convert a list of texts to their corresponding vocabulary indexes.
    '''
    if use_mask:
        out = -np.ones((len(texts), dim), dtype=np.int32)
        mask = np.zeros((len(texts), dim), dtype=np.float32)
    else:
        out = -2 * np.ones((len(texts), dim), dtype=np.int32)

    for i, text in enumerate(texts):
        for j, symbol in enumerate(text[:dim]):
            if symbol in vocab:
                out[i,j] = vocab[symbol]
            else:
                out[i,j] = -1 # for UNKnown symbols

        if use_mask:
            mask[i,:j] = 1.

    if use_mask:
        return out, mask
    else:
        return out



def text2idx2(texts, vocab, dim, use_mask=False):
    '''
    Convert a list of texts to their corresponding vocabulary indexes.
    '''
    
    if use_mask:
        out = -np.ones((len(texts), dim), dtype=np.int32)
        mask = np.zeros((len(texts), dim), dtype=np.float32)
    else:
        out = -2 * np.ones((len(texts), dim), dtype=np.int32)

    out_lst = []
    for i, text in enumerate(texts):
        words = wordpunct_tokenize(text)[:dim]

        for j, word in enumerate(words):
            if word in vocab:
                out[i,j] = vocab[word]
            else:
                out[i,j] = -1 # Unknown words

        out_lst.append(words)

        if use_mask:
            mask[i,:j] = 1.

    if use_mask:
        return out, mask, out_lst
    else:
        return out, out_lst


def idx2text(idxs, vocabinv, max_words=-1, char=False, output_unk=True):
    '''
    Convert list of vocabulary indexes to text.
    '''
    out = []
    for i in idxs:
        if i >= 0:
            out.append(vocabinv[i])
        elif i == -1:
            if output_unk:
                out.append('<UNK>')
        else:
            break

        if max_words > -1:
            if len(out) >= max_words:
                break

    if char:
        return ''.join(out)
    else:
        return ' '.join(out)


def n_words(words, vocab):
    '''
    Counts the number of words that have an entry in the vocabulary.
    '''
    c = 0
    for word in words:
        if word in vocab:
            c += 1
    return c


def load_vocab(path, n_words=None):
    dic = pkl.load(open(path, "rb"))
    vocab = {}

    if not n_words:
        n_words = len(dic.keys())

    for i, word in enumerate(dic.keys()[:n_words]):
        vocab[word] = i
    return vocab




