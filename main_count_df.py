'''
    Create a pandas dataframe of n-gram vocabulary and dump into pickle file. The dataframe contain (1) token,
    (2) document frequency, (3) frequency ranking
    :argument
        doc file name: A file containing documents, each of which is stored in json format with the following keys,
        (1) "title", (2) "desc", (3) "tag"
        key: Key index of data to be process
        dataframe file name: The name of the file into which the dataframe is to be dumped.
        number of pool processes.
        number of n-gram - default = 5.
        shutdown: [bool] shutdown after complete - default = False
'''

import src.dataimporter as di
import src.tokenizer as tt
import tltk
from multiprocessing import Pool


def wrapper_tokenize(text):
    return tt.tokenize(text, tltk_tokenize, ngram, './Dict/charset', cleaner)


def tltk_tokenize(text):
    ret = tltk.segment(text).replace('<u/>', '').replace('<s/>', '').split('|')
    return ret

def simple_split(token_text):
    '''input a string of word-tokens separated by "|": return list of word-tokens'''
    ret = token_text.split('|')
    return set(ret)


if __name__ == '__main__':

    import sys
    import pandas
    import os
    from tqdm import tqdm

    # declare arguments
    # data20180620_test.json
    doc_filename = sys.argv[1]
    key = sys.argv[2]
    out_filename = sys.argv[3]
    try:
        pool_process = int(sys.argv[4])
    except:
        pool_process = 8
    try:
        ngram = int(sys.argv[5])
    except:
        ngram = 5
    try:
        shutDown = sys.argv[6]
    except:
        shutDown = False

    # data import
    print('Loading data from ' + doc_filename)
    data_df = di.dataImporter(doc_filename)
    data = data_df[key]
    print('Successfully load data from ' + doc_filename + '\n')

    # multiprocessing : tokenize
    cleaner = tt.cleanerFactory("./Resource/charset")

    # multiprocessing : tokenize docs
    print('    Tokenizing data - pool processes = ' + str(pool_process))

    pbar = tqdm(total=int(len(data)))
    doc_tokens = []
    with Pool(pool_process) as pool:
        pool_result = pool.imap(wrapper_tokenize, data, chunksize=100)
        for item in pool_result:
            doc_tokens.append(set(simple_split(item)))
            pbar.update()
    pbar.close()
    # create fitted tfidf tokenizer for description docs
    print('    Couning document frequencies')
    doc_freq = {}
    for doc in doc_tokens:
        for token in doc:
            try:
                doc_freq[token] += 1
            except:
                doc_freq[token] = 1
    # create dataframe
    print('    Creating dataframe')
    data_input = [(item, doc_freq[item]) for item in doc_freq.keys()]
    doc_freq_df = pandas.DataFrame(data_input, columns=["token", "frequency"])
    # doc_freq_df.to_html('./test.html')
    doc_freq_df = doc_freq_df.sort_values(by=['frequency', 'token'], ascending=[False, True])
    doc_freq_df['rank'] = pandas.Series(list(range(1, len(data_input)+1)))
    print('    Saving output')
    doc_freq_df.to_pickle(out_filename)
    if shutDown:
        os.system('shutdown now -h')
