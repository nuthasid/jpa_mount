def dataImporter(filename):
    '''Input filename,json as string, return list of strings'''

    import json
    import pandas as pd
    try :
        datamat = pd.DataFrame(columns=["title", "desc", "tag"])
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_dict = json.loads(line, encoding='utf-8')
                datamat = datamat.append(line_dict, ignore_index=True)
    except:
        raise
    return datamat
