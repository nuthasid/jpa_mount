def dataImporter(filename):
    '''Input filename,json as string, return list of strings'''

    import json
    import pandas as pd
    import sys
    try :
        datamat = pd.DataFrame(columns=["title", "desc", "tag"])
        with open(filename, 'r', encoding='utf-8') as fin:
            counter = 1
            for line in fin:
                line_dict = json.loads(line, encoding='utf-8')
                datamat = datamat.append(line_dict, ignore_index=True)
                print('Loading doc: ' + str(counter), end='\r')
                sys.stdout.flush()
            print('')
    except:
        raise
    return datamat
