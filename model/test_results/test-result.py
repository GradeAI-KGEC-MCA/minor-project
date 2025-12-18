import json

def get_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def check_result(data):
    count_incorrect = 0
    count_correct = 0
    total_correct = 0
    total_incorrect = 0

    for record in data:
        if record['original_feedback'] == 'incorrect': total_incorrect+=1
        if record['original_feedback'] == 'correct': total_correct += 1
        if record['original_feedback'] == record['model_feedback'] and record['original_feedback'] == 'incorrect': count_incorrect+=1
        if record['original_feedback'] == record['model_feedback'] and record['original_feedback'] == 'correct': count_correct+=1

    
    print('**************************************************************Incorrect**************************************************************')
    print(f'Accuracy: [{count_incorrect}/{total_incorrect}]', count_incorrect/total_incorrect)
    print('**************************************************************Correct**************************************************************')
    print(f'Accuracy: [{count_correct}/{total_correct}]', count_correct/total_correct)
    print('**************************************************************Total**************************************************************')
    print(f'total records: {len(data)}')

print('==============================================================Unseen Answers==============================================================')
check_result(get_json('./model/test_results/unseen_a.json'))
print('==============================================================Unseen Questions==============================================================')
check_result(get_json('./model/test_results/unseen_q.json'))
