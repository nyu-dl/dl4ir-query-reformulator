import os
from collections import OrderedDict


######################
# General parameters #
######################
data_folder = '.'
n_words = 374000 # words for the vocabulary
vocab_path = data_folder + '/D_cbow_pdw_8B_norm.pkl' # Path to the python dictionary containing the vocabulary.
wordemb_path = data_folder + '/D_cbow_pdw_8B_norm.pkl' # Path to the python dictionary containing the word embeddings.
dataset_path = data_folder + '/jeopardy_dataset.hdf5' # path to load the hdf5 dataset containing queries and ground-truth documents.
docs_path = data_folder + '/jeopardy_corpus.hdf5' # Path to load the articles and links.
docs_path_term = data_folder + '/jeopardy_corpus.hdf5' # Path to load the articles and links.



############################
# Search Engine Parameters #
############################
engine = 'lucene' # Search engine used to retrieve documents.
n_threads = 20 # number of parallel process that will execute the queries on the search engine.
index_name = 'index' # index name for the search engine. Used when engine is 'lucene'.
index_name_term = 'index_terms' # index name for the search engine. Used when engine is 'lucene'.
index_folder = data_folder + '/' + index_name + '/' # folder to store lucene's index. It will be created in case it does not exist.
index_folder_term = data_folder + '/' + index_name_term + '/' # folder to store lucene's index. It will be created in case it does not exist.
local_index_folder = './' + index_name
local_index_folder_term = './' + index_name_term
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
dropout = -1 # If >0, <dropout> fraction of the units in the fully connected layers will be set to zero at training time.
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
max_feedback_docs_train = 1 # maximum number of feedback documents whose words be used to reformulate the query. Only used during training.
n_iterations = 2 # number of query reformulation iterations.
frozen_until = 1 # don't learn and act greedly until this iteration (inclusive). If frozen_until <= 0, learn everything.
reward = 'RECALL' # metric that will be optimized. Valid values are 'RECALL', 'F1', 'MAP', and 'gMAP'.
metrics_map = OrderedDict([('RECALL',0), ('PRECISION',1), ('F1',2), ('MAP',3), ('LOG-GMAP',4)])
q_0_fixed_until = 2 # Original query will be fixed until this iteration (inclusive). If <=0, original query can be modified in all iterations.
