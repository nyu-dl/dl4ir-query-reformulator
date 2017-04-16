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


def top_tfidf(txt, idf, N, min_freq=0):
    '''
    Return a list of top-N words indexes with respect to the TF-IDF.
    '''
    words = wordpunct_tokenize(txt)
    wf = {}
    for i, word in enumerate(words):
        if word in idf:
            if word not in wf:
                wf[word] = 0.
            wf[word] += 1.

    wf_ = {}
    for word, val in wf.items():
        if val >= min_freq:
            wf_[word] = (1 + math.log(val)) * idf[word] 
    wf = wf_

    keys, vals = wf.keys(), wf.values()
    keys2 = []
    vals2 = []
    for idx in np.argsort(np.asarray(vals))[::-1][:N]:
        keys2.append(keys[idx])
        vals2.append(vals[idx])
    return keys2, vals2


def load_synonyms():
    dic_thes = {}
    with open(prm.path_thes_dat, 'rb') as f:
        data = f.read().lower()
    header = 0
    with open(prm.path_thes_idx, 'rb') as f:
        for line in f:
            if header < 2:
                header += 1
                continue
            word_idx = line.rstrip().split("|")
            word, idx = word_idx[0], word_idx[1]
            idx = int(idx)
            j=0
            desc = ""
            while data[idx+j] != "\n":
                desc += data[idx+j]
                j += 1
            word_numlines = desc.rstrip().split("|")
            word, numlines = word_numlines[0], word_numlines[1]
            numlines = int(numlines)
            dic_thes[word] = []
            k = 0
            desc = ""
            while True:
                j += 1 
                desc += data[idx+j]
                if data[idx+j] == "\n":
                    k += 1
                    synonyms = desc.rstrip().split("|")[1:] #do not consider the first word because it refers to the POS tagging
                    dic_thes[word].extend(synonyms) #extend list of synonyms
                    desc = "" #start a new line
                if k == numlines:
                    break
    return dic_thes


def augment(texts, dic_thes):
    if prm.aug<2:
        return texts

    out = []
    for text in texts:

        words_orig = wordpunct_tokenize(text)
        maxrep = max(2,int(0.1*len(words_orig))) #define how many words will be replaced. For now, leave the maximum number as 10% of the words
        
        for j in range(prm.aug):
            words = list(words_orig) #copy
            for k in range(randint(1,maxrep)):
                idx = randint(0,len(words)-1)
                word = words[idx]
                if word in dic_thes:
                    
                    synonym = min(np.random.geometric(0.5), len(dic_thes[word])-1) #chose the synonym based on a geometric distribution
                    words[idx] = dic_thes[word][synonym]

            out.append(" ".join(words))

    return out


def remove_stop_words(words, stop):
    '''
    Remove stop words.
    If words is string, return string. If list, return list
    '''

    is_str = False
    if isinstance(words, basestring):
        is_str = True
        words = wordpunct_tokenize(words)

    words = filter(lambda word: word not in stop, words)

    if is_str:
        return ' '.join(words)
    else:
        return words




