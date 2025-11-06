import json
import random
folders = {
    'incorrect': './data/updated/incorrect/',
    'correct': './data/updated/correct/'
}
file_names = [
    'train',
    'validation',
    'unseen_questions',
    'unseen_answers'
]

def quicksort(data, low=0, high=None):
    if high is None:
        high = len(data) - 1
    if low < high:
        pivot_index = partition(data, low, high)
        quicksort(data, low, pivot_index - 1)
        quicksort(data, pivot_index + 1, high)
    return data

def partition(data, low, high):
    pivot = data[high]["question"]
    i = low - 1
    for j in range(low, high):
        if data[j]["question"] <= pivot:
            i += 1
            data[i], data[j] = data[j], data[i]
    data[i + 1], data[high] = data[high], data[i + 1]
    return i + 1

def find_max_scores(data):
    max_scores = {}
    for record in data:
        if record['question'] in max_scores: continue
        if record['verification_feedback'].lower() == 'correct':
            max_scores[record['question']] = record['score']
    return max_scores

def add_max_scores(data):
    max_scores = find_max_scores(data)
    for record in data:
        record['max_score'] = max_scores[record['question']]

def normalize_score(data):
    for record in data:
        record['normalized_score'] = round(record['score'] / record['max_score'], 2)
        record.pop('score', None)
    return data

def remove_partial_correct_data(data):
    binary_data = []
    for record in data:
        if record['verification_feedback'].lower() in ['correct', 'incorrect']:
            binary_data.append(record)
    
    return binary_data

def separate_correct_incorrect(data):
    correct = []
    partially_correct = []
    incorrect = []
    for record in data:
        if record['verification_feedback'].lower() == 'correct':
            correct.append(record)
        elif record['verification_feedback'].lower() == 'incorrect':
            incorrect.append(record)
        else:
            partially_correct.append(record)
    
    return {'correct':correct, 'incorrect': incorrect, 'partially_correct': partially_correct}

def save_json(data, path):
    inp = input('Do you want to save train.json? [Y/n]: ').lower()
    if inp in ['', 'y', 'yes']:
        with open(path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4)
        print('train.json saved!')
    else: print('Operation Terminated')
    

def get_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def count_data(data):
    count = 0

    for _ in data:
        count += 1
    
    return count

def get_questions(data):
    questions = {}
    for record in data:
        if record['question'] not in questions:
            questions[record['question']] = {
                'question': record['question'],
                'reference_answer': record['reference_answer'],
                'max_score': record['max_score']
                }
    
    return list(questions.values())

def combine_data(*args):
    combined = []
    for lst in args:
        combined.extend(lst)
    random.shuffle(combined)
    print('combined size: ', count_data(combined))
    return combined

def format_data(data: list[dict]) -> list[dict]:
    formatted: list[dict] = []

    for record in data:
        formatted.append({
            'question': record['question'],
            'reference_answer': record['reference_answer'],
            'provided_answer': record['provided_answer'],
            'answer_feedback': record['answer_feedback'],
            'verification_feedback': record['verification_feedback'],
            'max_score': record['max_score'],
            'normalized_score': record['normalized_score']
        })

    print(f'count: {count_data(formatted)}')
    return formatted

for file in file_names:
    print(file, ':\n=========================================================')
    correct_data = format_data(get_json(folders['correct']+file+'.json'))
    incorrect_data = format_data(get_json(folders['incorrect']+file+'.json'))

    data = combine_data(correct_data, incorrect_data)

    save_json(data, f'./data/updated/combined/{file}.json')

    print(count_data(data))

