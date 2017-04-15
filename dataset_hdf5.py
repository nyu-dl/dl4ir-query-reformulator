'''
Class to access queries and references stored in the hdf5 file.
'''
import h5py
import ast

class DatasetHDF5():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')


    def get_queries(self, dset=['train', 'valid', 'test']):
        '''
        Return the queries.
        'dset': list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for dname in dset:
            outs.append(list(self.f['queries_'+dname]))

        return outs


    def get_doc_ids(self, dset=['train', 'valid', 'test']):
        '''
        Return the <queries, references> pairs.
        'dset': list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for dname in dset:
            outs.append(map(ast.literal_eval, list(self.f['doc_ids_'+dname])))

        return outs

