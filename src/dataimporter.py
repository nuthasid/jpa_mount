def dataImporter(filename):
    '''Input filename,json as string, return list of strings'''

    import json
    import pandas as pd
    from tqdm import tqdm
    # import sys
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
                # if counter % 1000 == 0:
                    # print('Loading doc: ' + str(counter), end='\r')
                    # sys.stdout.flush()
            pbar.close()
    except:
        raise
    datamat = pd.DataFrame(data, columns=["title", "desc", "tag"])
    return datamat
