import json
import random
folders = [
    './data/updated/incorrect/',
    './data/updated/correct/'
]
file_names = [
    train
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
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)

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
    for key, lst in args.items():
        combined.extend(lst)
    random.shuffle(combined)
    return combined


for file in file_paths:
    # path = './data/updated/partially_correct/' + file + '.json' # +file+'.json'
    paths = './data/augmented_incorrect.json'
    with open(path, 'r', encoding='utf8') as f:
        data1 = json.load(f)
    
    path = './data/updated/incorrect/train.json'
    with open(path, 'r', encoding='utf8') as f:
        data2 = json.load(f)

    data = combine_data(data1, data2)
    
    path = './data/updated/incorrect/augmented_train.json'   
    save_json(data, path)
    print(count_data(data))


    
