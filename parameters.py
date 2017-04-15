import os
from collections import OrderedDict


######################
# General parameters #
######################
data_folder = '.'
n_words = 374000 # words for the vocabulary
vocab_path = data_folder + '/data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the vocabulary.
wordemb_path = data_folder + '/data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the word embeddings.
dataset_path = data_folder + '/data/jeopardy_dataset.hdf5' # path to load the hdf5 dataset containing queries and ground-truth documents.
docs_path = data_folder + '/data/wiki_628.hdf5' # Path to load the articles and links.
docs_path_term = data_folder + '/data/wiki_628.hdf5' # Path to load the articles and links.



############################
# Search Engine Parameters #
############################
engine = 'lucene' # Search engine used to retrieve documents.
n_threads = 20 # number of parallel process that will execute the queries on the search engine.
index_name = 'index_1276' # index name for the search engine. Used when engine is 'lucene'.
index_name_term = 'index_1276' # index name for the search engine. Used when engine is 'lucene'.
index_folder = data_folder + '/data/' + index_name + '/' # folder to store lucene's index. It will be created in case it does not exist.
index_folder_term = data_folder + '/data/' + index_name_term + '/' # folder to store lucene's index. It will be created in case it does not exist.
local_index_folder = './' + index_name
local_index_folder_term = './' + index_name_term
idf_path = None # data_folder + '/data/idf_1147.pkl' # path to save/load the idf dictionary.
use_cache = False # If True, cache (query-retrieved docs) pairs. Watch for memory usage.



####################
# Model parameters #
####################
optimizer='adam' # valid options are: 'sgd', 'rmsprop', 'adadelta', and 'adam'.
dim_proj=500  # LSTM number of hidden units.
dim_emb=500  # word embedding dimension.
patience=1000  # Number of epochs to wait before early stop if no progress.
max_epochs=5000  # The maximum number of epochs to run.
dispFreq=100 # Display to stdout the training progress every N updates.
lrate=0.0002  # Learning rate for sgd (not used for adadelta and rmsprop).
erate=0.002 # multiplier for the entropy regularization.
l2reg=0.0 # multiplier for the L2 regularization.
saveto='model.npz'  # The best model will be saved there.
validFreq=10000  # Compute the validation error after this number of updates.
saveFreq=10000  # Save the parameters after every saveFreq updates.
batch_size_train=64  # The batch size during training.
batch_size_pred=16 # The batch size during training.
#reload_model='model.npz'  # Path to a saved model we want to start from.
reload_model=False  # Path to a saved model we want to start from.
train_size=10000 # If >0, we keep only this number of train examples when measuring accuracy.
valid_size=10000 # If >0, we keep only this number of valid examples when measuring accuracy.
test_size=10000 # If >0, we keep only this number of test examples when measuring accuracy.
fixed_wemb = True # set to true if you don't want to learn the word embedding weights.
dropout = -1 # If >0, <dropout> fraction of the units in the non-recurrent layers will be set to zero at training time.
window_query = [3,3] # Window size for the CNN used on the query.
filters_query = [250,250] # Number of filters for the CNN used on the query.
window_cand = [9,3] # Window size for the CNN used on the candidate words.
filters_cand = [250,250] # Number of filters for the CNN used on the candidate words.
n_hidden_actor = [250] # number of hidden units per scoring layer on the actor.
n_hidden_critic = [250] # number of hidden units per scoring layer on the critic.
max_words_input = 200 # Maximum number of words from the input text.
max_terms_per_doc = 200 # Maximum number of candidate terms from each feedback doc. Must be always less than max_words_input .
max_candidates = 40 # maximum number of candidate documents that will be returned by the search engine.
max_feedback_docs = 7 # maximum number of feedback documents whose words be used to reformulate the query.
max_feedback_docs_train = 1 # maximum number of feedback documents whose words be used to reformulate the query. Only used during training. If supervised learning, this parameter must be equal to max_feedback_docs.
n_iterations = 2 # number of query reformulation iterations.
frozen_until = 1 # don't learn and act greedly until this iteration (inclusive). If frozen_until <= 0, learn everything.
reward = 'RECALL' # metric that will be optimized. Valid values are 'RECALL', 'F1', 'MAP', and 'gMAP'.
metrics_map = OrderedDict([('RECALL',0), ('PRECISION',1), ('F1',2), ('MAP',3), ('LOG-GMAP',4)])
top_tfidf = -1 # if > 0 , use top-<max_words_input> tf*idf words from the returned docs. If <= 0, use the first <max_words_input> words of the returned docs. This parameter must be less or equal than <max_words_input>.
idf_threshold = 0.0 # only words with IDF higher than this threshold will be used. For reference, a threshold of 4, corresponds to filter out the lowest 1349 words with respect to their idfs.
min_term_freq = 2 # if > 0, minimum number of times that a term must occur in the document to be considered as a top-tfidf term. Only used if top_tfidf > 0. 
state_encoder='NULL' # Valid values are 'LSTM', 'FF', 'NULL'. If 'NULL', previously selected words are not used to compute the new ones. If 'FF', previously selected words are used to compute new ones. If 'LSTM', the hidden state is used to select new words. The hidden state is computed based on input query, previously selected words, and previous hidden state.
cand_terms_source = 'doc' # Source where candidate terms are be drawn from. if 'syn', return wordnet synonyms. If 'synemb', return synonyms from word2vec cosine similarity. If 'doc', return terms from feedback documents. if 'all', return synomyms from all types.
syns_per_word = 1 # maximum number of synonyms per word in original query. Only used when return_words = 'syn', 'synemb' or 'all'.
synemb_path = data_folder + '/data/synemb.pkl' # path to save the dictionary of synonyms computed from word embeddings cosine similarity.
att_query = False # if True, use attention on query conditioned on the candidate word and hidden state. If False, the query is representation as the average word embeddings. Only used when state_encoder='LSTM'.
supervised_reload = '' # data_folder + '/data/supervised_743.hdf5' # Path to load the hdf5 containing the Ground-Truth labels for each candidate word. If None or False, use REINFORCE.
supervised = None # data_folder + '/data/supervised_1176.hdf5' # Path to save the hdf5 containing the Ground-Truth labels for each candidate word. If None or False, use REINFORCE.
supervised_threshold = 5e-3 # only label the word as positive if it adds a gain percentually larger than this threshold.
q_0_fixed_until = 2 # Original query will be fixed until this iteration (inclusive). If <=0, original query can be modified in all iterations.
