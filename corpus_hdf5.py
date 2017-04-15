'''
Class to access documents and links stored in the hdf5 file.
'''
import h5py

class CorpusHDF5():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')


    def get_article_text(self, article_id):
        return self.f['text'][article_id]
 

    def get_article_title(self, article_id):
        return self.f['title'][article_id]


    def get_titles_pos(self):
        '''
        Return a dictionary where the keys are articles' titles and the values are their offset in the data array.
        '''
        return dict((el,i) for i,el in enumerate(self.f['title'].value))


    def get_pos_titles(self):
        '''
        Return a dictionary where the keys are the articles' offset in the data array and the values are their titles.
        '''
        return dict((i,el) for i,el in enumerate(self.f['title'].value))


    def get_text_iter(self):
        return self.f['text']


    def get_title_iter(self):
        return self.f['title']
