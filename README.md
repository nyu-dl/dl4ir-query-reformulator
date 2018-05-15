# QueryReformulator

A Query Reformulation Framework based on Deep Reinforcement Learning.

- [Paper: Task-Oriented Query Reformulation with Reinforcement Learning](https://arxiv.org/abs/1704.04572)

- [Slides](https://github.com/nyu-dl/QueryReformulator/blob/master/Slides.pdf)

## Dataset and Other Files

The datasets and auxiliary files can be [downloaded here](https://drive.google.com/drive/folders/0BwmD_VLjROrfLWk3QmctMXpWRkE?usp=sharing). They are under BSD 3 License.

* **msa_dataset.hdf5**: MS Academic dataset: a query is the title of a paper and the ground-truth documents are the papers cited within.
* **msa_corpus.hdf5**: MS Academic corpus: each document consists of a paper title and abstract.
* **jeopardy_dataset.hdf5**: Jeopardy dataset: queries are Jeopardy! TV Show questions and answers are the Wikipedia articles whose title is the answer.
* **jeopardy_corpus.hdf5**: Jeopardy Corpus: All the English Wikipedia Articles (5.9M documents).
* **trec-car_dataset.hdf5**: [TREC-CAR dataset](http://trec-car.cs.unh.edu/): a query is Wikipedia article title + a section within that article. Ground-truth documents are paragraphs within that section.
* **trec-car_corpus.hdf5**: TREC-CAR Corpus: Half of the English Wikipedia Paragraphs, except abstracts.
* **D_cbow_pdw_8B_norm.pkl**: A python dictionary containing 374,000 pretrained word embeddings from the [Word2Vec tool](https://code.google.com/archive/p/word2vec/).

## Accessing the Dataset

The datasets are stored in the HDF5 format.

We provide wrapper classes to access them: dataset_hdf5.py and corpus_hdf5.py

The queries and documents can be accessed using the Python code below (h5py package is required):

```
#get training, validation and test lists of queries and relevant documents:

import dataset_hdf5
dt = dataset_hdf5.DatasetHDF5('path/to/the/dataset.hdf5')

queries_train, queries_valid, queries_test = dt.get_queries()
doc_titles_train, doc_titles_valid, doc_titles_test = dt.get_doc_ids()


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

## Training times

Each minibatch iteration should take approximately 1 second on a K80 GPU. It should take 800,000 iterations (7-10 days) to reach a Recall@40 of 47.6% in the TREC-CAR dataset. It is normal that the model starts to select terms only after iteration 50,000.


## Dependencies

To run the code, you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [Theano 0.9](http://deeplearning.net/software/theano/)
* [NLTK](http://www.nltk.org/)
* [h5py](http://www.h5py.org/)
* [PyLucene 6.2 or higher](http://lucene.apache.org/pylucene/)

We recommend that you have at least 32GB of RAM. If you are going to use a GPU, the card must have at least 6GB.

Note: If you are using Theano 1.0 you will probably see a "NullTypeGradError". Switching back to Theano 0.9 fixes this problem.

## Reference

If you use this code as part of any published research, please acknowledge the
following paper:

    @inproceedings{nogueira2017task,
      title={Task-Oriented Query Reformulation with Reinforcement Learning},
      author={Nogueira, Rodrigo and Cho, Kyunghyun},
      booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
      pages={574--583},
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
