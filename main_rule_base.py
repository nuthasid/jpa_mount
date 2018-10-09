"""
    Read json data file and txt rule files, compile matching, and export to html.
    :args data_file: json data file
    :args rule_file: txt rule file
    :args len_limit: number of job ads to display
    :args out_file: path to output html directory
    :opt: filter=: list of filters [filter=opt1,opt1,...]
"""


def read_rule(rules_file):

    def escapes(text):
        in_text = text[:]
        escapes_seq = (('|.', '\.'), ('|+', '\+'))
        for item_esc in escapes_seq:
            in_text = in_text.replace(item_esc[0], item_esc[1])
        return in_text

    import re
    import json
    with open(rules_file, 'r', encoding='utf-8') as fin:
        rule_list = [json_rule for json_rule in json.load(fin) if not json_rule['suppress']]
    rule_keys = ['title_inc', 'title_exc', 'desc_inc', 'desc_exc']
    # print(rule_list)
    rule_levels = []
    for index, _ in enumerate(rule_list):
        # print(rule_list[index])
        for rule_key in rule_keys:
            if rule_list[index][rule_key]:
                rule = escapes(rule_list[index][rule_key])
                rule = rule.split('||')
                rule_list[index][rule_key] = \
                    [re.compile(item_rule) for item_rule in rule]
            else:
                rule_list[index][rule_key] = None
        # print(rule_list[index])
        rule_list[index]['rule_no'] = index + 1
        rule_levels.append(rule_list[index]['level'])
        # print(rule_list[index])
    rule_levels = sorted(list(set(rule_levels)), reverse=True)
    return rule_list, rule_levels


def classify_doc(doc_in, rule_list, rule_levels):

    from copy import deepcopy

    doc_out = deepcopy(doc_in)
    doc_out['match'] = '000 ::'
    doc_out['predict_tag'] = 'NA'
    for level in rule_levels:
        rule_match = apply_rules(doc_in, rule_list, level)
        if rule_match['terminate']:
            doc_out['match'] = rule_match['match']
            doc_out['predict_tag'] = rule_match['predict_tag']
            return doc_out
        else:
            pass

    return doc_out


def apply_rules(doc_in, rule_list, level):

    rules_match = []
    ret = {'terminate': False, 'predict_tag': set(), 'match': []}
    rule_list = [rule for rule in rule_list if rule['level'] == level]
    for rule in rule_list:
        pattern_match = apply_pattern(rule['title_inc'], doc_in['title'], False) and \
                        apply_pattern(rule['desc_inc'], doc_in['desc'], False) and \
                        apply_pattern(rule['title_exc'], doc_in['title'], True) and \
                        apply_pattern(rule['desc_exc'], doc_in['desc'], True)
        if pattern_match:
            ret['predict_tag'].add(rule['tag'])
            rules_match.append(rule)
    if len(ret['predict_tag']) == 0:
        ret['terminate'] = False
    elif len(ret['predict_tag']) == 1:
        ret['terminate'] = True
        ret['predict_tag'] = '\n'.join(sorted(list(ret['predict_tag'])))
        ret['match'] = '222::\n' + str(rules_match[0]['rule_no'])
    else:
        ret['terminate'] = True
        ret['predict_tag'] = '\n'.join(sorted(list(ret['predict_tag'])))
        ret['match'] = '111:: ' + ', '.join([str(rule['rule_no'])
                                            for rule in rules_match])

    return ret


def apply_pattern(regex, text, reverse):

    if not regex:
        return True
    else:
        for rule in regex:
            if rule.search(text):
                pass
            else:
                return False if not reverse else True
        return True if not reverse else False


if __name__ == '__main__':

    import json
    import sys
    from src.export_html import PredictPage

    # parse arguments
    argvs = sys.argv[1:]
    if argvs[-1].find('filter') != -1:
        opt_filter = argvs.pop(-1)[7:].split(',')
    else:
        opt_filter = None
    data_file = argvs.pop(0)
    rule_file = argvs.pop(0)
    len_limit = int(argvs.pop(0))
    if argvs:
        out_dir = argvs.pop(0)
        if out_dir[-1] != '/':
            out_dir += '/'
    else:
        out_dir = './output/'

    # read data file (json format)
    with open(data_file, 'r', encoding='utf-8') as f:
        if opt_filter:
            data_load = json.load(f)
            data_set = []
            for item in data_load:
                for flt in opt_filter:
                    if item['title'].find(flt) != -1:
                        data_set.append(item)
                        break
        else:
            data_set = json.load(f)

            # read rule file
    rules, levels = read_rule(rule_file)

    # apply rules
    docs_not_found = []
    docs_conflict = []
    docs_unique_rule = []
    for doc in data_set:
        doc_apply = classify_doc(doc, rules, levels)
        if doc_apply['match'][:3] == '000':
            docs_not_found.append(doc_apply)
        elif doc_apply['match'][:3] == '111':
            docs_conflict.append(doc_apply)
        else:
            docs_unique_rule.append(doc_apply)
    print('No match: ', len(docs_not_found))
    print('Unique match: ', len(docs_unique_rule))
    print('Rule conflict: ', len(docs_conflict))

    # export to html
    html_not_found = PredictPage(title='Job Ads without Known Rule', page_title='Not Found',
                                 data=docs_not_found, len_limit=len_limit,
                                 data_head=['title', 'desc', 'predict_tag', 'match'])
    html_not_found.export_html(out_dir + '[0]rule_not_found.html')
    html_conflict = PredictPage(title='Job Ads with Conflicting Rules', page_title='Conflict',
                                data=docs_conflict, len_limit=len_limit,
                                data_head=['title', 'desc', 'predict_tag', 'match'])
    html_conflict.export_html(out_dir + '[1]conflict.html')
    html_unique_rule = PredictPage(title='Job Ads with a Unique Rule', page_title='Unique',
                                   data=docs_unique_rule, len_limit=len_limit,
                                   data_head=['title', 'desc', 'predict_tag', 'match'])
    html_unique_rule.export_html(out_dir + '[2]unique_rule.html')
