def dataImporter(filename):
    """Input filename,json as string, return list of strings"""

    import json
    import pandas as pd
    from tqdm import tqdm

    try:
        data = []
        with open(filename, 'r', encoding='utf-8') as fin:
            # counter = 0
            lines = fin.readlines()
            pbar = tqdm(total=len(lines))
            for line in lines:
                line_dict = json.loads(line, encoding='utf-8')
                data.append((line_dict['title'], line_dict['desc'], line_dict['tag']))
                pbar.update()
            pbar.close()
    except FileNotFoundError:
        raise
    datamat = pd.DataFrame(data, columns=["title", "desc", "tag"])
    return datamat


class DataController:
    """
        DataController object contain documents in pandas.DataFrame object. Initiating DataController automatically
        tokenize documents. DataController contain several methods to maintain and expand dataset.
    """

    def __init__(self, documents='null', data_format='null', num_pool=30, chunk=10, add_column=None):
        """
        Initialize DataController object.
        :param documents: (default='null') Documents source to be loaded into the object. It can be (1) path to a text
        file containing json documents with the following keys (title, desc, tag) (2) path to a pickle file containing
        pandas.DataFrame object, (3) a pandas.DataFrame object. If 'null', the Data Controller will initialize with
        blank data.
        :param data_format: (str, default='null') Documents source format. Use 'f_json' for json documents contained in
        a text file. Use 'f_df' for a pickle file containing documents stored in pandas.DataFrame. Otherwise leave
        default='null'.
        :param num_pool: (int, default=30) number of parallel pool processes.
        :param chunk: (int, default=100) number of jobs per batch in each parallel pool processes.
        :param add_column: (str or list, default=None) Additional columns (keys) to be included in dataset.
        """
        import json
        import pandas as pd
        from multiprocessing import Pool
        from tqdm import tqdm

        #  Set class parameters
        self.num_pool = num_pool
        self.chunk_size = chunk

        # define data keys
        dat_columns = ['job_id', 'title', 'desc', 'tag',
                       'title_seg', 'desc_seg', 'sampled']
        # add user specified keys
        if add_column:
            if type(add_column) is list:  # if user provide list of keys
                dat_columns += add_column
            elif type(add_column) is str:  # if user provide a single key
                dat_columns += [add_column]
            else:
                raise TypeError('add_column must be str or list.')

        if data_format == 'f_json':  # if user specify json data format
            print('loading data')
            # load and parse each json formatted documents and store into (list)loaded_data.
            with open(documents, 'rt', encoding='utf-8') as fin:
                loaded_data = json.load(fin)

            docs_tokens = []
            print('pre-processing')
            pbar = tqdm(total=int(len(loaded_data)))  # create progress bar.
            with Pool(self.num_pool) as pool:  # init parallel pool processes.
                # use _pre_process function to clean, lemmatize, tokenize
                # and generate job_id using hash function for each documents.
                pool_result = pool.imap(self._pre_process, loaded_data, self.chunk_size)
                for item in pool_result:
                    docs_tokens.append(item)
                    pbar.update()
            pbar.close()

            # create object dataset in pandas.DataFrame format.
            self.dataSet = pd.DataFrame(docs_tokens, columns=dat_columns)

        elif data_format == 'f_df':  # if user specify pickle formatted file containing pandas.DataFrame.
            self.dataSet = pd.read_pickle(documents)  # load DataFrame from pickle file and assign to object dataset.
        elif type(documents) is pd.DataFrame:  # if user provide pandas.DataFrame as an input parameter.
            self.dataSet = documents  # simple assign DataFrame to object dataset.
        elif data_format == 'null':  # if user specify null data.
            print('DataController created with blank data')
            self.dataSet = pd.DataFrame(columns=dat_columns)  # assign blank DataFrame to object dataset.
        else:
            raise TypeError('Invalid data/file object provided.')

    def re_init(self):
        """
        Re-process object-document. Tasks includes (1) reset indexes, (2) clean, lemmatize, and tokenize all documents,
        (3) reset sampling marker.
        :return: None
        """

        import pandas as pd
        from multiprocessing import Pool
        from tqdm import tqdm
        from copy import deepcopy

        loaded_data = []
        # Define data keys.
        columns = ['job_id', 'title', 'desc', 'tag', 'title_seg', 'desc_seg', 'sampled']
        data_buffer = deepcopy(self.dataSet)  # make a copy of object-document to avoid side effect.
        data_buffer = data_buffer.reset_index(drop=True)  # reset indexes.
        for index, _, in data_buffer.iterrows():
            # convert each document entry in object-document into dict format in order to make them applicable to
            # self._pre_process function.
            data_line = {'title': data_buffer.loc[index, 'title'], 'desc': data_buffer.loc[index, 'desc'],
                         'tag': data_buffer.loc[index, 'tag']}
            # Discard any document with extremely low information content, i.e. documents with fewer than 60 characters.
            if len(data_line['desc']) > 60:  # should make "60" threshold adjustable.
                loaded_data.append(data_line)

        docs_tokens = []
        print('re-initialize data set: pre-processing')
        pbar = tqdm(total=int(len(loaded_data)))  # create progress bar.
        with Pool(self.num_pool) as pool:
            # use _pre_process function to clean, lemmatize, tokenize
            # and generate job_id using hash function for each documents.
            pool_result = pool.imap(self._pre_process, loaded_data, chunksize=self.chunk_size)
            for item in pool_result:
                docs_tokens.append(item)
                pbar.update()
        pbar.close()

        # create object dataset in pandas.DataFrame format.
        self.dataSet = pd.DataFrame(docs_tokens, columns=columns)

    def get_training_set(self, label_class):
        """
        Create training set from a specified label (tag).
        :param label_class: (str) Target label (or tag) of documents on which the classifier will be trained.
        :return: pandas.DataFrame
        """
        import pandas as pd

        # target_dataset is a set of document with target label (tag = label_class).
        target_dataset = self.dataSet[self.dataSet['tag'] == label_class]
        rest_dataset = self.dataSet[self.dataSet['tag'] != label_class]  # the rest of documents.

        if target_dataset.shape[0] < rest_dataset.shape[0]:  # target documents has less items than the rest
            # concatenate the collection of ALL TARGET DOCUMENTS with an equally sized collection of a subset of the
            # OFF-LABEL documents.
            training_dataset = pd.concat([target_dataset, rest_dataset.sample(n=target_dataset.shape[0])])
        else:
            # concatenate the collection of ALL OFF-LABEL DOCUMENTS with an equally sized collection of a subset of the
            # target documents.
            training_dataset = pd.concat([target_dataset.sample(n=rest_dataset.shape[0]), rest_dataset])
        # shuffle data using sample fraction = 1
        training_dataset = training_dataset.sample(frac=1)
        #  change tag of all sample not tagged label_tag
        training_dataset.loc[training_dataset.tag != label_class, 'tag'] = '!' + label_class
        return training_dataset

    def sample(self, size='ALL', repeated=False, record_selection=False):
        """
        Create a sample documents from object-documents.
        :param size: (int ot float, default='ALL') Sample size parameter. Either in term of absolute sample size (int)
        or a fraction of object-document collection (float). 'ALL' will return the entire collection.
        :param repeated: (boolean) True if the sample is to be drawn from the entire object-document collection. False
        if the sampling is to exclude previously drawn documents, i.e. documents marked sampled=True.
        :param record_selection: (boolean) True will mark selected document as sampled=True, False will leave sampled
        mark unchanged.
        :return: pandas.DataFrame
        """

        if repeated:  # if repeated
            sample_source = self.dataSet  # assign entire object-document collection to sample_source
        else:  # if not repeated
            sample_source = self.dataSet.loc[not self.dataSet['sampled']]  # assigned un-sampled documents to source
        if size == 'ALL':
            return self.dataSet  # if ALL return entire collection
        elif type(size) is float:  # if sample size is defined in term of fraction
            size = int(self.dataSet.shape[0] * size)  # calculate absolute sample size
        size = min(size, sample_source.shape[0])  # calculate the minimum of absolute sample size and the size of source
        ret_sample = sample_source.sample(n=size)  # sampling
        if record_selection:
            self.dataSet.loc[ret_sample.index, 'sampled'] = True  # mark sampled if record_selection=True
        return ret_sample

    def join(self, data_objects):
        """
        (Method) Joint (concat) two or more DataController objects to self.
        :param data_objects: DataController objects
        :return: None (method)
        """

        def _check_type(data_controller):
            """
            check if data objects are dataController object(s).
            :param data_controller: DataController object or a list of DataController.
            :return: list of of dataSet if the dataSet is DataController object, raise error if otherwise.
            """

            data_obj = deepcopy(data_controller)  # make a copy of the collection of DataControllers

            if type(data_obj) is list or type(data_obj) is tuple:
                counter = 0
                for item in data_obj:
                    ret = type(item) is DataController
                    if not ret:
                        raise TypeError('Data object (' + str(counter) + ') in data list is not DataController.')
                    counter += 1
                return tuple(data_obj)
            elif type(data_obj) is DataController:
                return tuple([data_obj])
            else:
                raise TypeError('Data object must be DataController object or a list of DataController.')

        def _merge_data(data_objs):
            """
            Merge multiple pandas.DataFrame objects to self.dataSet.
            :param data_objs: pandas.DataFrame objects containing no fields (columns) other than that presented in the
            self.dataSet.
            :return: None
            """

            import pandas

            merged = pandas.DataFrame()  # Declare new DataFrame
            counter = 1
            for data_obj in data_objs:
                if not set(list(data_obj)) - data_fields:  # Check if the DataFrame contains foreign columns.
                    raise ValueError('Data object (' + str(counter) + ') contain foreign columns(s). Undo join.')
                else:
                    print('Merging dataset (' + str(counter) + ') with keys: ' + str(set(list(data_obj))))
                    merged = pandas.concat(objs=(merged, data_obj), ignore_index=True)
                counter += 1
            return merged

        from copy import deepcopy

        dataset = deepcopy(data_objects)
        data_fields = set(list(self.dataSet))
        dataset = _check_type(dataset)  # Check if data_objects are DataController and collect dataSet from objects.
        dataset = _merge_data(dataset)
        self.dataSet = self.dataSet.append(dataset, ignore_index=True)

    def add_column(self, columns):
        """
        Add columns to self.dataSet
        :param columns: {'key_1':[values],...,'key_n:[values]}
        :return: None
        """
        for key in sorted(list(columns.keys())):
            self.dataSet[key] = columns[key]

    @staticmethod
    def read_pickle(pickle_file):
        """
        Load DataController from pickle file
        :param pickle_file: (str) path to pickle file containing DataController object.
        :return: DataController object
        """

        import pickle

        with open(pickle_file, 'rb') as f:
            ret = pickle.load(f)
        return ret

    @staticmethod
    def _pre_process(doc_dict):
        """
        Pre-process a document data; 1) generate id, 2) tokenize the document.
        :param doc_dict: A document stored in a dictionary object with the following keys; 'title', 'desc', 'tag'.
        :return: A list of tokenized keys; job_id, title, desc, tag, title_seg, desc_seg, False.
        """
        from src import tokenizer
        import tltk
        import hashlib
        import time

        def tltk_tokenize(text):
            ret = tltk.segment(text).replace('<u/>', '').replace('<s/>', '').split('|')
            return ret

        cleaner = tokenizer.cleaner_generator('../Resource/charset')
        title = doc_dict['title']
        desc = doc_dict['desc']
        title_seg = tokenizer.tokenize(title, cleaner, tltk_tokenize, 5)
        desc_seg = tokenizer.tokenize(desc, cleaner, tltk_tokenize, 5)
        tag = doc_dict['tag']
        in_str = str(time.time()) + title + desc
        job_id = hashlib.md5(bytes(in_str, 'utf-8')).hexdigest()
        return [job_id, title, desc, tag, title_seg, desc_seg, False]

    def set_param(self, num_pool=30, chunk_size=100):
        """
        Set class parameters.
        :param num_pool: Number of parallel pools.
        :param chunk_size: Number of jobs per chunk in each parallel pools.
        :return: None
        """
        self.num_pool = num_pool
        self.chunk_size = chunk_size
