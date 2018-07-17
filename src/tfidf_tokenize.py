def tfidf_tokenize(token_texts, max_df=1.0, min_df=1):
    '''
        input a list of word-tokens, each separated by "|": return fitted tfidf vectorizer.
        parameter:  max_df=(0,1.0)
                        if is float: ignore tokens that are present in more than max_df% of the documents;
                        if is int: ignore tokens that are present in more than max_df documents.
                    min_df=(0m1.0)
                        if is float: ignore tokens that are present in less than min_df% of the documents;
                        if is int: ignore tokens that are present in less than min_df documents.
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorize = TfidfVectorizer(tokenizer=simple_split, max_df=max_df, min_df=min_df)
    tfidf_vectorize.fit(token_texts)
    return tfidf_vectorize


def simple_split(token_text):
    '''input a string of word-tokens separated by "|": return list of word-tokens'''
    ret = token_text.split('|')
    return ret
