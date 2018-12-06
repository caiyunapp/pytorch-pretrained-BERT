from collections import OrderedDict
import re
import json
import os


def score():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WSC_child_problem.json') 
    with open(path, 'r') as f:
        data_l = json.load(f)
    f.close()

    result = []
    for data in data_l:
        if data['sentences'] != []:
            for s in data['sentences']:
                score = 0
                if s['predict_answer'] != []:
                    predict_answer = eval(s['predict_answer'][0])
                    if re.findall(predict_answer[0], str(s['correct_answer']), flags=re.IGNORECASE):
                        score = 1
                s['score'] = score
            res = {'index': data['index'], 's': s}
            result.append(res)

    total_score = 0
    for r in result:
        total_score += r['s']['score']
    print('Score each valid problems:')
    description = ' Total valid problems: {0}\n Correct answers: {1}\n Acuracy: {2}'
    print(description.format(len(result), total_score, float(total_score/len(result))))

    print()
    result_dict = {}
    for r in result:
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
            result_dict[index].append(r)
        else:
            result_dict[index] = [r]

    total_score = 0
    for key in result_dict.keys():
        score = 1 
        for r in result_dict[key]: 
            if r['s']['score'] == 0:
                score = 0
        total_score += score
    print('Score each valid problem groups:')
    description = ' Total valid problems: {0}\n Correct answers: {1}\n Acuracy: {2}'
    print(description.format(len(result_dict), total_score, float(total_score/len(result_dict))))


if __name__ == '__main__':
    score()
