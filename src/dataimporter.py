def dataImporter(filename):
    '''Input filename,json as string, return list of strings'''

    import json
    import pandas as pd
    from tqdm import tqdm
    # import sys
    try:
        datamat = pd.DataFrame(columns=["title", "desc", "tag"])
        with open(filename, 'r', encoding='utf-8') as fin:
            # counter = 0
            lines = fin.readlines()
            pbar = tqdm(total=len(lines))
            for line in lines:
                line_dict = json.loads(line, encoding='utf-8')
                datamat = datamat.append(line_dict, ignore_index=True)
                pbar.update()
                # if counter % 1000 == 0:
                    # print('Loading doc: ' + str(counter), end='\r')
                    # sys.stdout.flush()
            pbar.close()
    except:
        raise
    return datamat
