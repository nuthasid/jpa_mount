'''
    Read pandas.DataFrame from pickle file and display each item user can key new tag
    Input:  path to pickle file
            path to new DataFrame file
            default tag name (def=tag_name)
            positive tag (pos=tag_name)
            negative tag (neg=tag_name)
            [optional] filter (filter=tag)
            [optional] range (range=begin:end)
            [optional] sampling (sampling=sample_size(int or float))
'''

if __name__ == '__main__':

    import pandas
    import sys

    # parse arguments
    argvs = sys.argv[1:]
    data_file = argvs.pop(0)
    ret_file = argvs.pop(0)
    kwargs = {'filter': None, 'range': {'begin': None, 'End': None}}
    tags = {'def': None, '-': None, '+': None}
    sampling = 1.0
    for index in reversed(range(len(argvs))):
        if argvs[index].find('filter') != -1:
            kwargs['filter'] = argvs[index].split('=')[1]
        if argvs[index].find('range') != -1:
            dat_range = argvs[index].split('=')[1]
            kwargs['range']['begin'] = int(dat_range.split(':')[0]) if dat_range.split(':')[0] else None
            kwargs['range']['end'] = int(dat_range.split(':')[1]) if dat_range.split(':')[1] else None
        if argvs[index].find('neg') != -1:
            tags['-'] = argvs[index].split('=')[1]
        if argvs[index].find('pos') != -1:
            tags['+'] = argvs[index].split('=')[1]
        if argvs[index].find('def') != -1:
            tags['def'] = argvs[index].split('=')[1]
        if argvs[index].find('sampling') != -1:
            sampling = argvs[index].split('=')[1]
            try:
                sampling = int(sampling)
            except ValueError:
                sampling = float(sampling)
        argvs.pop(index)

    dataSet = pandas.read_pickle(data_file)
    print('\n\n==== Sample data ====')
    print(dataSet.iloc[:4])
    print('Total observations: ' + str(dataSet.count()))
    print('=' * 21)
    print('\n' * 2)
    dataSet = dataSet.iloc[kwargs['range']['begin']:kwargs['range']['end']]
    dataSet = dataSet.loc[dataSet['predict'] == kwargs['filter']]
    if type(sampling) is int:
        newDataSet = dataSet.sampling(n=sampling)
    elif type(sampling) is float:
        newDataSet = dataSet.sample(frac=sampling)
    else:
        raise ValueError('invalid sampling parameter')
    newDataSet = newDataSet.reset_index(drop=True)
    print('\n\n==== Tagging data ====')
    print(newDataSet.iloc[:4])
    print('Total observations: ' + str(newDataSet.count()))
    print('=' * 21)
    print('\n' * 4)
    for index, _ in newDataSet.iterrows():
        print('\n' * 8)
        print('current loc: ', index)
        print('\ntitle: \n', newDataSet.iloc[index]['title'])
        print('\ndescription: \n', newDataSet.iloc[index]['desc'])
        print('\n' * 8)
        tag_recorded = False
        while not tag_recorded:
            inp = input('>>>')
            if inp == '':
                newDataSet.at[index, 'tag'] = tags['def']
                print('Entry tagged as ' + tags['def'])
                print(newDataSet.iloc[index])
                tag_recorded = True
            elif inp in ['-', '+']:
                newDataSet.at[index, 'tag'] = tags[inp]
                print('Entry tagged as ' + tags[inp])
                print(newDataSet.iloc[index])
                tag_recorded = True
            else:
                print('tag must be "-" for ' + tags['-'] + ' tag or "+" for ' + tags['+'] + ' tag.')
    newDataSet.to_pickle(ret_file)
