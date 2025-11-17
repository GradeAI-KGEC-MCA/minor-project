import json
import random
folders = {
    'incorrect': './data/updated/incorrect/',
    'correct': './data/updated/correct/',
    'combined': './data/updated/combined/',
    'partially_correct': './data/updated/partially_correct/',
}
file_names = [
    'train',
    'validation',
    'unseen_questions',
    'unseen_answers'
]

original_path = 'data/original/'

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

    return data

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
    inp = input('Do you want to save? [Y/n]: ').lower()
    if inp in ['', 'y', 'yes']:
        with open(path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print('data saved!')
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

def remove_id(data):
    for record in data:
        record.pop('id', None)

    return data

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
            'id': record['id'],
            'question': record['question'].lower(),
            'reference_answer': record['reference_answer'].lower(),
            'provided_answer': record['provided_answer'].lower(),
            'answer_feedback': record['answer_feedback'].lower(),
            'verification_feedback': record['verification_feedback'].lower(),
            'max_score': record['max_score'],
            'normalized_score': record['normalized_score']
        })

    print(f'count: {count_data(formatted)}')
    return lower_case_data(formatted)

def lower_case_data(data):
    for record in data:
        for key, value in record.items():
            if isinstance(value, str):
                record[key] = value.lower()
    return data

print(__name__, '\n\n')
if __name__ == '__main__':

    for file in file_names:
        data = get_json(original_path+file+'.json')
        data = quicksort(data)
        data = add_max_scores(data)
        data = normalize_score(data)
        data = format_data(data)
        # data = remove_id(data)
        data = separate_correct_incorrect(data)

        save_json(data['correct'], folders['correct']+file+'.json')
        save_json(data['incorrect'], folders['incorrect']+file+'.json')
        save_json(data['partially_correct'], folders['partially_correct']+file+'.json')