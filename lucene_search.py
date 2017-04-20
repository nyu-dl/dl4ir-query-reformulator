# -*- coding: utf-8 -*-
'''
Use Lucene to retrieve candidate documents for given a query.
'''
import shutil
import os
import lucene
import parameters as prm
import utils
import itertools
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory, NIOFSDirectory, MMapDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher, MatchAllDocsQuery, BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser
import time
from collections import OrderedDict, defaultdict
from multiprocessing.pool import ThreadPool
import Queue
import math
from nltk.tokenize import wordpunct_tokenize
import cPickle as pkl


class LuceneSearch():

    def __init__(self):

        self.env = lucene.initVM(initialheap='28g', maxheap='28g', vmargs=['-Djava.awt.headless=true'])
        self.vocab = None

        BooleanQuery.setMaxClauseCount(2048)

        if not os.path.exists(prm.index_folder):
            print 'Creating index at', prm.index_folder
            if prm.docs_path == prm.docs_path_term:
                add_terms = True
            else:
                add_terms = False
            self.create_index(prm.index_folder, prm.docs_path, add_terms)

        if prm.local_index_folder:
            print 'copying index from', prm.index_folder, 'to', prm.local_index_folder
            if os.path.exists(prm.local_index_folder):
                print 'Folder', prm.local_index_folder, 'already exists! Doing nothing.'
            else:
                shutil.copytree(prm.index_folder, prm.local_index_folder)
            self.index_folder = prm.local_index_folder
        else:
            self.index_folder = prm.index_folder

        fsDir = MMapDirectory(Paths.get(prm.index_folder))
        self.searcher = IndexSearcher(DirectoryReader.open(fsDir))

        if prm.docs_path != prm.docs_path_term:
            if not os.path.exists(prm.index_folder_term):
                print 'Creating index at', prm.index_folder_term
                self.create_index(prm.index_folder_term, prm.docs_path_term, add_terms=True)

            if prm.local_index_folder_term:
                print 'copying index from', prm.index_folder_term, 'to', prm.local_index_folder_term
                if os.path.exists(prm.local_index_folder_term):
                    print 'Folder', prm.local_index_folder_term, 'already exists! Doing nothing.'
                else:
                    shutil.copytree(prm.index_folder_term, prm.local_index_folder_term)
                self.index_folder_term = prm.local_index_folder_term
            else:
                self.index_folder_term = prm.index_folder_term
            fsDir_term = MMapDirectory(Paths.get(prm.index_folder_term))
            self.searcher_term = IndexSearcher(DirectoryReader.open(fsDir_term))

        self.analyzer = StandardAnalyzer()
        self.pool = ThreadPool(processes=prm.n_threads)
        self.cache = {}
        
        print 'Loading Title-ID mapping...'
        self.title_id_map, self.id_title_map = self.get_title_id_map()

    def get_title_id_map(self):

        # get number of docs
        n_docs = self.searcher.getIndexReader().numDocs()

        title_id = {}
        id_title = {}
        query = MatchAllDocsQuery()
        hits = self.searcher.search(query, n_docs)
        for hit in hits.scoreDocs:
            doc = self.searcher.doc(hit.doc)
            idd = int(doc['id'])
            title = doc['title']
            title_id[title] = idd
            id_title[idd] = title

        return title_id, id_title


    def add_doc(self, doc_id, title, txt, add_terms):

        doc = Document()
        txt = utils.clean(txt)

        if add_terms:
            txt_ = txt.lower()
            words_idx, words = utils.text2idx2([txt_], self.vocab, prm.max_terms_per_doc)
            words_idx = words_idx[0]
            words = words[0]

        doc.add(Field("id", str(doc_id), self.t1))
        doc.add(Field("title", title, self.t1))
        doc.add(Field("text", txt, self.t2))
        if add_terms:
            doc.add(Field("word_idx", ' '.join(map(str,words_idx)), self.t3))
            doc.add(Field("word", '<&>'.join(words), self.t3))
        self.writer.addDocument(doc)


    def create_index(self, index_folder, docs_path, add_terms=False):

        print 'Loading Vocab...'
        if not self.vocab:
            self.vocab = utils.load_vocab(prm.vocab_path, prm.n_words)

        os.mkdir(index_folder)

        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setIndexOptions(IndexOptions.DOCS)

        self.t2 = FieldType()
        self.t2.setStored(False)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        self.t3 = FieldType()
        self.t3.setStored(True)
        self.t3.setIndexOptions(IndexOptions.NONE)
       
        fsDir = MMapDirectory(Paths.get(index_folder))
        writerConfig = IndexWriterConfig(StandardAnalyzer())
        self.writer = IndexWriter(fsDir, writerConfig)
        print "%d docs in index" % self.writer.numDocs()
        print "Indexing documents..."

        doc_id = 0

        import corpus_hdf5
        corpus = corpus_hdf5.CorpusHDF5(docs_path) 
        for txt in corpus.get_text_iter():
            title = corpus.get_article_title(doc_id)
            self.add_doc(doc_id, title, txt, add_terms)
            if doc_id % 1000 == 0:
                print 'indexing doc', doc_id
            doc_id += 1
                 
        print "Index of %d docs..." % self.writer.numDocs()
        self.writer.close()


    def search_multithread(self, qs, max_cand, max_full_cand, searcher):

        self.max_cand = max_cand
        self.max_full_cand = max_full_cand
        self.curr_searcher = searcher
        out = self.pool.map(self.search_multithread_part, qs)
 
        return out


    def search_multithread_part(self, q):

        if not self.env.isCurrentThreadAttached():
            self.env.attachCurrentThread()
    
        if q in self.cache:
            return self.cache[q]
        else:

            try:
                q = q.replace('AND','\\AND').replace('OR','\\OR').replace('NOT','\\NOT')
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
            except:
                print 'Unexpected error when processing query:', str(q)
                print 'Using query "dummy".'
                q = 'dummy'
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))

            c = OrderedDict()
            hits = self.curr_searcher.search(query, self.max_cand)

            for i, hit in enumerate(hits.scoreDocs):
                doc = self.curr_searcher.doc(hit.doc)
                if i < self.max_full_cand:
                    word_idx = map(int, doc['word_idx'].split(' '))
                    word = doc['word'].split('<&>')
                else:
                    word_idx = []
                    word = []
                c[int(doc['id'])] = [word_idx, word]

            return c

    
    def search_singlethread(self, qs, max_cand, max_full_cand, curr_searcher):

        out = []
        for q in qs:
            if q in self.cache:
                out.append(self.cache[q])
            else:
                try:
                    q = q.replace('AND','\\AND').replace('OR','\\OR').replace('NOT','\\NOT')
                    query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
                except:
                    print 'Unexpected error when processing query:', str(q)
                    print 'Using query "dummy".'
                    query = QueryParser("text", self.analyzer).parse(QueryParser.escape('dummy'))

                c = OrderedDict()
                hits = curr_searcher.search(query, max_cand)

                for i, hit in enumerate(hits.scoreDocs):
                    doc = curr_searcher.doc(hit.doc)
                    if i < max_full_cand:
                        word_idx = map(int, doc['word_idx'].split(' '))
                        word = doc['word'].split('<&>')
                    else:
                        word_idx = []
                        word = []
                    c[int(doc['id'])] = [word_idx, word]

                out.append(c)

        return out


    def get_candidates(self, qs, max_cand, max_full_cand=None, save_cache=False, extra_terms=True):
        if not max_full_cand:
            max_full_cand = max_cand

        if prm.docs_path != prm.docs_path_term:
            max_cand2 = 0
        else:
            max_cand2 = max_full_cand
        if prm.n_threads > 1:
            out = self.search_multithread(qs, max_cand, max_cand2, self.searcher)
            if (prm.docs_path != prm.docs_path_term) and extra_terms:
                terms = self.search_multithread(qs, max_full_cand, max_full_cand, self.searcher_term)
        else:
            out = self.search_singlethread(qs, max_cand, max_cand2, self.searcher)
            if (prm.docs_path != prm.docs_path_term) and extra_terms:
                terms = self.search_singlethread(qs, max_full_cand, max_full_cand, self.searcher_term)

        if (prm.docs_path != prm.docs_path_term) and extra_terms:
            for outt, termss in itertools.izip(out, terms):                
                for cand_id, term in itertools.izip(outt.keys()[:max_full_cand], termss.values()):
                    outt[cand_id] = term
  
        if save_cache:
            for q, c in itertools.izip(qs, out):
                if q not in self.cache:
                    self.cache[q] = c

        return out
