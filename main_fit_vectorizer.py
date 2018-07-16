'''
    Create an object of fitted tfidf vectorizer and dump into a pickle file.
    :argument
        doc file name: A file containing documents, each of which is stored in json format with the following keys,
        (1) "title", (2) "desc", (3) "tag"
        vectorizer file name: The name of the file into which vectorizer object is to be dumped.
        number of pool processes.
        number of n-gram for title tokenizer - default = 4.
        number of n-gram for description tokenizer - default = 2.
'''

import src.dataimporter as di
import src.tokenizer as tt
from src.tfidf_tokenize import tfidf_tokenize as tfidf
import tltk
from multiprocessing import Pool
from tqdm import tqdm


def wrapper_tokenize_desc(text):
    return tt.tokenize(text, tltk_tokenize, desc_ngram, './Dict/charset', cleaner)


def wrapper_tokenize_title(text):
    return tt.tokenize(text, tltk_tokenize, title_ngram, './Dict/charset', cleaner)


def tltk_tokenize(text):
    ret = tltk.segment(text).split('|')
    return ret


class TFIDF_Vectorizer:
    '''
    Create an object of tfidf vectorizer for both job title docs and job description doc.
    Attribute:
        vectorize_title: tfidf vectorizer for job title.
        vectorize_desc: tfidf vectorizer for job description.
        title_para: parameters of job title vectorizer - {"max_df":max_df, "min_df":min_df}.
        desc_para: parameters of job description vectorizer - {"max_df":max_df, "min_df":min_df}.
        filename: name of the file into which the object is to be dumped.
    '''

    def __init__(self, vectitle, vecdesc, filename):
        self.vectorize_title = vectitle
        self.title_para = {'max_df': vectitle.max_df, 'min_df': vectitle.min_df}
        self.vectorize_desc = vecdesc
        self.desc_para = {'max_df': vecdesc.max_df, 'min_df': vecdesc.min_df}
        self.filename = filename

    def dump(self):
        with open(self.filename, 'wb') as file_out:
            pickle.dump(self, file_out)


if __name__ == '__main__':

    import sys
    import pickle
    from tqdm import tqdm

    # declare arguments
    # data20180620_test.json
    doc_filename = sys.argv[1]
    out_filename = sys.argv[2]
    try:
        pool_process = int(sys.argv[3])
    except:
        pool_process = 8
    try:
        title_ngram = int(sys.argv[4])
    except:
        title_ngram = 4
    try:
        desc_ngram = int(sys.argv[5])
    except:
        desc_ngram = 2

    # data import
    print('Loading data from ' + doc_filename)
    data_df = di.dataImporter(doc_filename)
    desc_data = data_df['desc']
    title_data = data_df['title']
    print('Successfully load data from ' + doc_filename + '\n')

    # multiprocessing : tokenize
    cleaner = tt.cleanerFactory("./Resource/charset")

    # multiprocessing : tokenize job description docs
    print('Fitting job description vectorizer')
    print('    Tokenizing job description')

    pbar = tqdm(total=int(len(desc_data) / 2))
    desc_tokens = []
    with Pool(pool_process) as pool:
        pool_result = pool.imap(wrapper_tokenize_desc, desc_data)
        count = 0
        for item in pool_result:
            desc_tokens.append(item)
            count += 1
            if count % 2 == 0:
                pbar.update()
    pbar.close()
    # create fitted tfidf tokenizer for description docs
    print('    fitting')
    tfidf_desc = tfidf(desc_tokens, max_df=0.95, min_df=0.005)
    print('Completed fitting job description vectorizer\n')

    # multiprocessing : tokenize job title docs
    print('Fitting job title vectorizer')
    print('    Tokenizing title description')
    pbar = tqdm(total=int(len(title_data) / 2))
    title_tokens = []
    with Pool(pool_process) as pool:
        pool_result = pool.imap(wrapper_tokenize_title, title_data)
        count = 0
        for item in pool_result:
            title_tokens.append(item)
            count += 1
            if count % 2 == 0:
                pbar.update()
    pbar.close()
    # create fitted tfidf tokenizer for title docs
    print('    fitting')
    tfidf_title = tfidf(title_tokens, max_df=0.95, min_df=0.005)
    print('Completed fitting job title vectorizer\n')

    print('Creating fitted vectorizer object')
    # create fitted vectorizer object
    tfidf_vectorizer = TFIDF_Vectorizer(tfidf_title, tfidf_desc, out_filename)
    # dump fitted vectorizer object into a file
    tfidf_vectorizer.dump()

    print('Vectorizer is created and dumped into ' + out_filename)

    # map(tt.tokenize(), data_list)
