
class Classifier:

    def __init__(self, classifier, vectorizer):
        """
        Init Classifier
        :param classifier: A classifier object with 'predict' method.
        :param vectorizer: A vectorizier object wth 'transform' method.
        """

        self.classifier = classifier
        self.vectorizer = vectorizer

    def predict(self, document):
        """
        Predict the label of the document(s).
        :param document: pandas.DataFrame objects containing segmented documents, e.g. with columns 'title_seg' and
        'desc_seg'.
        :return: pandas.DataFrame object containing document(s) and predicted labels.
        """

        from copy import deepcopy
        from scipy.sparse import hstack
        import pandas

        sample = deepcopy(document)
        desc = sample['desc_seg']
        title = sample['title_seg']

        # transform-vectorize
        title_vec = self.vectorizer.vectorize_title.transform(title)
        desc_vec = self.vectorizer.vectorize_desc.transform(desc)
        # stack title onto desc
        data_vec = hstack([title_vec, desc_vec])
        # predict class
        label_predict = self.classifier.predict(data_vec)
        # add prediction to pandas.DataFrame
        prediction = deepcopy(document)
        prediction['predict'] = pandas.Series(label_predict, index=prediction.index)

        return prediction

    def save_pickle(self, pickle_file):
        """
        Save Classifier from pickle file
        :param pickle_file: (str) path to pickle file containing Classifier object.
        :return: None
        """

        import pickle

        with open(pickle_file, 'rb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read_pickle(pickle_file):
        """
        Load Classifier from pickle file
        :param pickle_file: (str) path to pickle file containing Classifier object.
        :return: Classifier object
        """

        import pickle

        with open(pickle_file, 'rb') as f:
            ret = pickle.load(f)
        return ret
