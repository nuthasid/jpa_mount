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
        import pickle
        with open(self.filename, 'wb') as file_out:
            pickle.dump(self, file_out)
