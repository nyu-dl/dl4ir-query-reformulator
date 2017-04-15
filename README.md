# WebNav

A Query Reformulation Framework based on Deep Reinforcement Learning.

Link to the paper: [Task-Oriented Query Reformulation with Reinforcement Learning]()


## Dataset and Other Files

The datasets and auxiliary files can be [downloaded here](https://drive.google.com/drive/folders/0BwmD_VLjROrfLWk3QmctMXpWRkE?usp=sharing):

* **msa_dataset.hdf5**: MS Academic dataset: a query is the title of a paper and the ground-truth documents are the papers cited within.
* **msa_corpus.hdf5**: MS Academic corpus: each document consists of a paper title and abstract.
* **jeopardy_dataset.hdf5**: Jeopardy dataset: queries are Jeopardy! TV Show questions and answers are the Wikipedia articles whose title is the answer.
* **jeopardy_corpus.hdf5**: Jeopardy Corpus: All the English Wikipedia Articles (5.9M documents).
* **trec-car_dataset.hdf5**: [TREC-CAR dataset](http://trec-car.cs.unh.edu/): a query is Wikipedia article title + a section within that article. Ground-truth documents are paragraphs within that section.
* **trec-car_corpus.hdf5**: TREC-CAR Corpus: Half of the English Wikipedia Paragraphs (3.5M), except abstracts.
* **D_cbow_pdw_8B.pkl**: a python dictionary containing 374,000 words where the values are pretrained embeddings from ["Word2Vec tool"](https://code.google.com/archive/p/word2vec/).

## Accessing the Dataset

The datasets are stored in the HDF5 format.

We provide wrapper classes to access them: dataset_hdf5.py and corpus_hdf5.py

The queries and documents can be accessed using the Python code below (h5py package is required):

```
#get training, validation and test lists of queries and relevant documents:

import dataset_hdf5
dt = dataset_hdf5.DatasetHDF5('path/to/the/dataset.hdf5')

queries_train, queries_valid, queries_test = dt.get_queries()
doc_ids_train, doc_ids_valid, doc_ids_test = dt.get_doc_ids()


# iterate over all documents in the corpus:

import corpus_hdf5
cp = corpus_hdf5.CorpusHDF5('path/to/the/corpus.hdf5')

for i, text in enumerate(cp.get_text_iter()):
    print 'text:', text
    print 'title:', cp.get_article_title(i)
```


## Running the Model

After changing the properties in the parameters.py file to point to your local paths, the model can be trained using the following command:

```
THEANO_FLAGS='floatX=float32' python run.py
```

If you want to use a GPU:

```
THEANO_FLAGS='floatX=float32,device=gpu0' python run.py
```


##Dependencies

To run the code, you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [Theano 0.9 or higher](http://deeplearning.net/software/theano/)
* [NLTK](http://www.nltk.org/)
* [h5py](http://www.h5py.org/)
* [PyLucene 6.2 or higher](http://lucene.apache.org/pylucene/)

We recommend that you have at least 32GB of RAM. If you are going to use a GPU, the card must have at least 6GB.


## Reference

If you use this code as part of any published research, please acknowledge the
following paper:

    @article{x,
        title={Task-Oriented Query Reformulation with Reinforcement Learning},
        author={x},
        journal={x},
        year={2017}
    } 

## License

Copyright (c) 2017, Rodrigo Nogueira and Kyunghyun Cho

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither this project nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
