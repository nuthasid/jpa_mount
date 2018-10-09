def tokenize(document, cleaner, th_tokenizer, n_grams,
             stop_en_filename=None, stop_th_filename=None, keywords_filename=None):
    """
    Clean, tokenize, and generate n-gram from a document (string).
    :param document: String containing a document.
    :param th_tokenizer: Thai language tokenizer function returning a list of tokens.
    :param n_grams: n length of n-gram.
    :param cleaner: cleaner function.
    :param stop_en_filename: Path to txt file containing English stop word.
    :param stop_th_filename: Path to txt file containing Thai stop word.
    :param keywords_filename: Path to txt file containing keywords.
    :return: String of tokens separated by '|'.
    """

    from copy import deepcopy

    document = deepcopy(document)  # make a copy of text.

    # load word lis from txt file.
    stopwords_en = get_word_list(stop_en_filename) if stop_en_filename else set()
    stopwords_th = get_word_list(stop_th_filename) if stop_th_filename else set()
    keywords = get_word_list(keywords_filename) if stop_th_filename else set()

    # clean text
    # (1) remove invalid characters/alphabets,
    # (2) split sentence merge,
    # (3) split adjunct English - Thai tokens,
    # (4) remove unuseful string pattern, e.g.names,
    # (5) split sentences joined by bullet markers
    document = cleaner(document)

    # tokenize document
    # (1) lemmatize English token excluding keywords
    # (2) segment words
    # (3) remove stopwords excluding keywords
    document = tokenize_cleaned(document, th_tokenizer, stopwords_en, stopwords_th, keywords)

    # merge token into one string whereas each tokens are separated by '|' and
    # sentence markers are designated by '|\\\\|'
    sentences = '|'.join(document)
    sentences = sentences.split('|\\\\|')  # split into list of sentences.

    for sentence in sentences:  # iterate over sentences.
        document.extend(n_grams_compile(sentence, n_grams))  # make n-grams and add to the document.

    document = '|'.join(document)  # merge all tokens into one string separated by '|' for further processing.

    return document


def tokenize_cleaned(document, th_tokenizer, stopwords_en, stopwords_th, keywords):
    """
    Tokenize and lemmatize tokens in document.
    :param document: Document in string
    :param th_tokenizer: Thai tokenizer function (return list)
    :param stopwords_en: set() of English stop word
    :param stopwords_th: set() of Thai stop word
    :param keywords: set() of keywords.
    :return: list of tokens.
    """
    from copy import deepcopy
    from nltk import WordNetLemmatizer
    import re

    def test_all_en_alpha(text):  # test if characters in the string are all English alphabet.
        roman_alpha = [chr(alpha) for alpha in range(65, 90)] + \
                      [chr(alpha) for alpha in range(97, 122)]
        for alpha in text:
            if alpha not in roman_alpha:
                return False
        return True

    pattern_thai_char = re.compile(u'[\u0e00-\u0e7f]')  # Thai alphabet pattern
    word_stem_func = WordNetLemmatizer()  # declare English lemmatizer.

    document = deepcopy(document)
    document = document.split(' ')  # split to form a list of phrases which are separated by '\s'
    # remove English stop word.
    document = [token for token in document
                if token not in stopwords_en]
    # Lemmatize English tokens.
    document = [word_stem_func.lemmatize(token)
                if test_all_en_alpha(token) and token not in keywords  # do not lemmatize keywords.
                else token
                for token in document]

    # tokenize Thai phrase.
    tokenized = []
    for token in document:
        if pattern_thai_char.search(token):  # check if phrase is in Thai
            tokenized.extend(th_tokenizer(token))  # extend to include a list of Thai tokens
        else:
            tokenized.append(token)  # append non-Thai tokens

    # remove Thai stop word
    for token_index in reversed(range(len(tokenized))):  # iterate backward
        if tokenized[token_index] in stopwords_th:  # if token is Thai stop word
            tokenized.pop(token_index)  # remove Thai stop word from doc

    return tokenized


def load_char_set(filename):
    """
    Load a set of legitimate characters from a text file.
    :param filename: Path to character set file.
    :return: A dict of characters with ord value.
    """
    charset = {}
    with open(filename, 'rt') as char_file:
        for char in char_file.read().split('\n'):
            char = char.replace('\n', '')
            if len(char) < 4:
                charset[char] = ord(char)
            else:
                charset[chr(int(char, 16))] = int(char, 16)
    return charset


def get_word_list(filename):
    """
    Load a word list from a text file.
    :param filename: Path to a text file containing word list (each words are separated by \n).
    :return: A set of words/tokens.
    """

    with open(filename, 'rt', encoding='utf-8') as fin:
        words_set = set([item for item in fin.read().split('\n')])
    return words_set


def n_gram_make(tokens, n):
    """
    Compile specific "n" n-grams from a list of tokens.
    :param tokens: A list of tokens.
    :param n: 'n' length of n-gram
    :return: A list of specific "n" n-grams - separated by '\s'.
    """

    ngram_ret = []
    for word_index, token in enumerate(tokens[:-(n - 1)]):  # iterate from first token to n-1 position.
        ngram_output = [token for token in tokens[word_index:word_index + n]]  # make n-gram
        ngram_ret.append(' '.join(ngram_output))  # append n-gram to list.

    return ngram_ret


def n_grams_compile(sentence, n):
    """
    Create a list of n-gram from n=2 to n=N.
    :param sentence: String containing sentence.
    :param n: maximum n number for n-gram.
    :return: A list of n-grams.
    """

    from copy import deepcopy

    sentence = deepcopy(sentence)
    # split running string and remove null token
    tokens = [token for token in sentence.split('|') if len(token) > 0]

    if n == 1:  # if n = 1  simply return None.
        return None
    else:
        n_tokens = []
        for grams in range(2, n + 1):  # iterate for generating n-gram of n=2 to n=N
            ngrams = n_gram_make(tokens, grams)  # for a given n=n, make n-gram.
            n_tokens.extend(ngrams)  # store resulting n-gram into a list.
        return n_tokens


def cleaner_generator(char_set_filename, keywords_filename=None):
    """
    create cleaner(text) function.
    :param char_set_filename: path to a text file containing a valid character set.
    :param keywords_filename: path to a text file special keywords.
    :return: cleaner(text) function.
    """

    def split_th_en(in_text, splitter):
        """
        Separate English text from Thai text.
        :param in_text: A string of input text.
        :param splitter: re.compile pattern indicating Thai character.
        :return: A string whereby English and Thai texts are separated by space.
        """

        from copy import deepcopy

        insert_pos = []
        in_text = deepcopy(in_text)
        for pos, item in enumerate(in_text[:-2]):
            # check is string start with Thai character and followed with English character and wise versa.
            if splitter.search(in_text[pos:pos + 2]):
                insert_pos.append(pos + 1)
        for pos in reversed(insert_pos):
            in_text = in_text[:pos] + ' ' + in_text[pos:]
        return in_text

    def split_sentence_merge(text, run_pattern, keyword):
        """
        Split end-of-sentence merge, separated with '\\\\'.
        :param text: Text to be split.
        :param run_pattern: Pattern of run-on sentence joints.
        :param keyword: Keywords that tend to be mistook for run-ons.
        :return: Text with sentences separated by '\\\\'.
        """

        from copy import deepcopy

        text = deepcopy(text)
        # Create a list of tokens by separating by '\s'.
        token_list = text.split(' ')  # a list of tokens.
        ret_string = ''
        for token in token_list:  # run through each token
            if run_pattern.search(text) and text not in keyword:  # if token match merged sentence.
                c_text = run_pattern.search(text)  # match group(0)
                # separate merged tokens by '\\\\'
                split_sentence = '\\\\'.join([c_text.string[:c_text.span()[0] + 1],
                                              c_text.string[c_text.span()[1] - 1:]])
            else:
                split_sentence = token  # if not merged tokens - do nothing.
            ret_string = ' '.join([ret_string, split_sentence])  # join split tokens with previous tokens.
        return ret_string

    def remove_invalid_char(val_text, char_pat, char_set):
        """
        Clean text of non-sense character5s/alphabets.
        :param val_text: String to be validated.
        :param char_pat: re.compile of character set to remove.
        :param char_set: Set of characters to include.
        :return: A string cleaned of non-sense character5s/alphabets.
        """

        from copy import deepcopy

        val_text = deepcopy(val_text)
        val_text = val_text.replace('&amp;', ' ')
        val_text = val_text.replace('&nbsp;', ' ')
        ret_text = ''
        for cha in val_text:
            if char_set.get(cha):
                ret_text += cha
        while char_pat.search(ret_text):
            ret_text = char_pat.sub(' ', ret_text)
        while ret_text.find('  ') != -1:
            ret_text = ret_text.replace('  ', ' ')
        return ret_text

    def cleaner(text):
        """
        Clean a document by
        (1) remove unwanted characters/alphabets,
        (2) remove unuseful string pattern, e.g. names,
        (3) split adjunct English-Thai tokens,
        (4) split sentences joined by bullet markers,
        (5) split merged sentences.
        :param text: a string document to be processed.
        :return: string of text whereby sentences are separated by '\\\\'.
        """
        import re

        # ===== BEGIN define pattern =====
        pattern_new_sentence = re.compile(r'\.[0-9]+[).]\s')  # new sentence with numbered bullet.
        # Thai - English switching.
        pattern_th_in = re.compile(u'([^\u0e00-\u0e7f][\u0e00-\u0e7f])|([\u0e00-\u0e7f][^\u0e00-\u0e7f])')
        pattern_phone_number = re.compile('[0-9\-]{9-12}')  # phone number
        pattern_email = re.compile('[a-zA-Z._\-0-9]+@[a-zA-Z._\-0-9]+')  # email pattern
        pattern_url = re.compile('(https://|www.)[a-zA-Z0-9]+.[a-z]+[^\s]*')  # url pattern
        pattern_thai_name = re.compile(u'\u0e04\u0e38\u0e13\s*[\u0e00-\u0e7f]+\s+')  # Thai name pattern
        pattern_sentence_merge = re.compile('[a-z][A-Z]')  # Sentence merged pattern.
        pattern_num_bullet = re.compile('^[0-9]+[).]*$')  # numbered bullet
        pattern_double_sentence_stop_maker = re.compile(r'(\\\\)(.){,2}(\\\\)')
        pattern_white_space = re.compile(r'(\s|\t|\n)+')
        pattern_thai_phrase_space = re.compile(u'[\u0e01-\u0e3a\u0e40-\u0e5d](\s)+[\u0e01-\u0e3a\u0e40-\u0e5d]')
        # discarded letters.
        pattern_garbage_lead_char = re.compile(r'^-|^\||^\.|^#{1,2}|^(-\|)|^(\+\|)|^(#\|)^(\.\|)')
        # ===== END ======

        charset = load_char_set(char_set_filename)  # load valid character set.
        keywords = get_word_list(keywords_filename) if keywords_filename else set()  # load keywords.

        # conversion table for thai number to arabic
        thai_num_list = zip([chr(i) for i in range(3664, 3674)], [' ' + str(i) + ' ' for i in range(0, 10)])
        for item in thai_num_list:  # convert thai number to arabic alphabet.
            re.sub(item[0], item[1], text)

        # begin replacing useless tokens.
        text = text.replace(u'\u0e46', ' ')
        text = pattern_email.sub(' ', text)
        text = pattern_url.sub(' ', text)
        text = pattern_phone_number.sub(' ', text)
        text = pattern_thai_name.sub(' ', text)
        text = pattern_white_space.sub(' ', text)
        text = pattern_thai_phrase_space.sub('\\\\', text)
        text = pattern_num_bullet.sub('\\\\', text)
        # End===================================

        text = split_th_en(text, pattern_th_in)  # split run-on English-Thai tokens.
        text = pattern_new_sentence.sub('\\\\', text)  # replace bullets with sentence marker
        text = text.replace('.', '\\\\')  # English sentences are separated by "\\\\".
        text = pattern_double_sentence_stop_maker.sub('\\\\', text)
        text = remove_invalid_char(text, pattern_garbage_lead_char, charset)  # Remove invalid characters.
        text = split_sentence_merge(text, pattern_sentence_merge, keywords)  # split sentence merged.

        return text

    return cleaner  # return cleaner(text) function to caller.
