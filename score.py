from collections import OrderedDict
import re
import json
import os
from pprint import pprint


def score():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WSC_child_problem.json') 
    with open(path, 'r') as f:
        data_l = json.load(f)
    f.close()

    result = []
    s_order = ['sentence', 'answer1', 'answer0', 'correct_answer', 'adjacent_ref', 'predict_answer', 'score']
    data_order = ['index', 'sentences']
    for data in data_l:
        if data['sentences'] != []:
            for i in range(len(data['sentences'])):
                s = data['sentences'][i]
                score = 0
                if s['predict_answer'] != []:
                    predict_answer = s['predict_answer'][0]
                    if any(answer.lower() == predict_answer[0] for answer in s['correct_answer']):
                        score = 1
                s['score'] = score
                s = OrderedDict(sorted(s.items(), key=lambda i:s_order.index(i[0])))
                data['sentences'][i] = s
        data = OrderedDict(sorted(data.items(), key=lambda i:data_order.index(i[0])))
        result.append(data)

    print('Save the score in WSC_child_problem.json\n')
    with open(path, 'w') as f:
         json.dump(result, f, indent=4, separators=(',', ': '), ensure_ascii=False)
    f.close()

    total_score = 0
    total_valid_problems = 0
    l = {}
    for r in result:
        for s in r['sentences']:
            if 'score' in s:
                total_valid_problems += 1
                score = s['score']
                total_score += score
            if r['index'] not in l.keys():
                l[r['index']] = [0, 1]
            else:
                l[r['index']][1] += 1
            if score == 1:
                l[r['index']][0] += 1
    print('Correct problems:')
    pprint(l)
    print()

    print('Score each valid problems:')
    description = ' Total valid problems: {0}\n Correct answers: {1}\n Accuracy: {2}'
    print(description.format(total_valid_problems, total_score, float(total_score/total_valid_problems)))

    print()
    result_dict = {}
    for r in result:
        for s in r['sentences']:
            if 'score' in s:
                index = r['index']
                if index < 252:
                    if index % 2 == 1:
                        index -= 1
                elif index in [252, 253, 254]:
                    index = 252
                else:
                    if index % 2 == 0:
                        index -= 1
                if index in result_dict.keys():
                    result_dict[index].append(s)
                else:
                    result_dict[index] = [s]

    total_score = 0
    for key in result_dict.keys():
        score = 1 
        for s in result_dict[key]:
            if s['score'] == 0:
                score = 0
        total_score += score
    print('Score each valid problem groups:')
    description = ' Total valid problems: {0}\n Correct answers: {1}\n Accuracy: {2}'
    print(description.format(len(result_dict), total_score, float(total_score/len(result_dict))))


if __name__ == '__main__':
    score()
