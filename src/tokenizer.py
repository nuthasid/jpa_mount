def tokenize(text, THTokenizer, ngrams, cleaner, charsetFilename=None, stop_enFilename=None, stop_thFilename=None, keywordsFilename=None):
    '''Input string of jobads, return string of tokens that are n-gramed and tokenized'''
    import re
    pattern_sentence_collide = re.compile('[a-z][A-Z]]')
    pattern_thai_char = re.compile(u'[\u0e00-\u0e7f]')

    cleanedText = cleaner(text)

    text_split = cleanedText.split('|')
    first_pass = firstPass(text_split, pattern_sentence_collide)
    second_pass = secondPass(first_pass, pattern_thai_char,
                             THTokenizer, ngrams)
    tokensString = "|".join(second_pass)
    return tokensString


def firstPass(textList, pattern, keyword=[]):
    '''Split runon EN sentences'''
    first_pass = []
    for i, item in enumerate(textList):
        if pattern.search(item) and item not in keyword:
            c_text = pattern.search(item)
            first_pass.extend([c_text.string[:c_text.span()[0] + 1], c_text.string[c_text.span()[1] - 1:]])
        else:
            first_pass.append(item)
    return first_pass


def secondPass(textList, pattern, THTokenizer, ngrams):
    '''Tokenize TH sentences and compile ngrams of TH words'''
    second_pass = []
    for i, chunk in enumerate(textList):
        if pattern.search(chunk) and len(chunk) > 1:
            ## Tokenize TH Chunk and do grams
            thtokensList = THTokenizer(chunk)
            ngrammedList = n_grams_compile(thtokensList, ngrams, pattern)
            second_pass.extend(ngrammedList)
        else:
            ## Lowercase for EN chunk
            second_pass.append(chunk.lower())
    return second_pass


def n_gram_compile(tokens, n, THPattern):
    '''Actually Do n grams'''
    tokens = tokens[:]
    tokensOutput = []
    for wordPos, token in enumerate(tokens[:-(n - 1)]):
        # Do grams until n-1 position
        new_token = ''
        for word in tokens[wordPos:wordPos + n]:
            if THPattern.search(word) and len(word) > 1:
                new_token += word
            else:
                new_token = ''
                break
        if new_token:
            # Found tokens
            tokensOutput.extend([new_token])
    return tokensOutput


def n_grams_compile(tokens, n, THPattern):
    '''Do from 2 grams to n grams'''
    if n == 1:
        return tokens
    else:
        n_tokens = []
        for grams in range(2, n + 1):
            ## Create list of grams from 2 grams to n grams
            ngrammed = n_gram_compile(tokens, grams, THPattern)
            n_tokens.extend(ngrammed)
        n_tokens = tokens + n_tokens
        return n_tokens


def cleanerFactory(charsetFilename, stop_enFilename=None, stop_thFilename=None, keywordsFilename=None):
    '''Create cleaner function'''
    def loadCharset(filename):
        charset = {}
        with open(filename, 'rt') as charfile:
            for char in charfile.read().split('\n'):
                char = char.replace('\n', '')
                if len(char) < 4:
                    charset[char] = ord(char)
                else:
                    charset[chr(int(char, 16))] = int(char, 16)
        return charset

    def getwords(filename):
        '''Create a set of words from filename'''
        with open(filename, 'rt', encoding='utf-8') as fin:
            wordsSet = set([item for item in fin.read().split('\n')])
        return  wordsSet

    def cleaner(text):
        def split_th_en(splt_text, splitter):
            insert_pos = []
            splt_text = splt_text[:]
            for pos, item in enumerate(splt_text[:-2]):
                if splitter.search(splt_text[pos:pos + 2]) or splitter.search(splt_text[pos:pos + 2]):
                    insert_pos.append(pos + 1)
            for pos in reversed(insert_pos):
                splt_text = splt_text[:pos] + ' ' + splt_text[pos:]
            return splt_text

        def validate_char(val_text, splitter, charset):
            val_text = val_text.replace('&amp;', ' ')
            val_text = val_text.replace('&nbsp;', ' ')
            ret_text = ''
            for cha in val_text:
                if charset.get(cha):
                    ret_text += cha
            while splitter.search(ret_text):
                ret_text = splitter.sub(' ', ret_text)
            while ret_text.find('  ') != -1:
                ret_text = ret_text.replace('  ', ' ')
            return ret_text

        def remove_thai_stop(th_text, stop_th):
            stop_pos = [[0, 0]]
            # TH : do longest matching
            for j in range(len(th_text) - 1):
                for k in range(j + 1, min(len(th_text), j + 36)):
                    if th_text[j:k] in stop_th:
                        # found keyword +++ instead of returning string - return positions that is
                        # i to j
                        if j <= stop_pos[-1][1]:
                            stop_pos[-10] = [stop_pos[-1][0], k]
                        else:
                            stop_pos.append([j, k])
                        break
            newstr = ''
            if len(stop_pos) == 1:
                newstr = th_text
            else:
                for j in range(len(stop_pos) - 1):
                    newstr += th_text[stop_pos[j][1]:stop_pos[j + 1][0]] + ' '
            return newstr

        import re
        from nltk.stem import WordNetLemmatizer
        pattern_new_sentence = re.compile('\.[0-9]+(\)|\.) ')
        pattern_th_in = re.compile(u'[^\u0e00-\u0e7f][\u0e00-\u0e7f]')
        pattern_num_bullet = re.compile('^[0-9]+(\)|\.)*$')
        pattern_eng_token = re.compile('^[a-zA-Z]+$')
        pattern_phone_number = re.compile('[0-9\-]{9-12}')
        pattern_email = re.compile('[a-zA-Z._\-0-9]+@[a-zA-Z._\-0-9]+')
        pattern_url = re.compile('(https://|www.)[a-zA-Z0-9]+.[a-z]+[^\s]*')
        pattern_thai_name = re.compile(u'\u0e04\u0e38\u0e13\s*[\u0e00-\u0e7f]+\s+')
        pattern_prefix_garbage = re.compile('^\-|^\||^\.|^\#{1,2}|^(\-\|)|^(\+\|)|^(\#\|)^(\.\|)')
        charset = loadCharset(charsetFilename)
        stemmer = WordNetLemmatizer()

        stopwords_EN = getwords(stop_enFilename) if stop_enFilename else set()
        stopwords_TH = getwords(stop_thFilename) if stop_thFilename else set()
        keywords = getwords(keywordsFilename) if keywordsFilename else set()

        text = text.replace(u'\u0e46', ' ')
        text = pattern_email.sub(' ', text)
        text = pattern_url.sub(' ', text)
        text = pattern_phone_number.sub(' ', text)
        text = pattern_thai_name.sub(' ', text)
        text = split_th_en(text, pattern_th_in)
        text = pattern_new_sentence.sub(' . ', text)
        text = text.replace('.', ' . ')
        text = validate_char(text, pattern_prefix_garbage, charset)
        text = remove_thai_stop(text, stopwords_TH)
        text_split = text.split(' ')
        text_split = [item for item in text_split[:] if item not in stopwords_EN and not pattern_num_bullet.search(item)]
        text_split = [stemmer.lemmatize(item) if pattern_eng_token.search(item) and item not in keywords else item for item in text_split[:]]
        text = '|'.join(text_split)
        return text

    # return cleaner function to caller
    return cleaner

