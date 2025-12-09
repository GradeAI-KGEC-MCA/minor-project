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
    # 'unseen_questions',
    'unseen_answers'
]

original_path = './data/original/'

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

def separate(data):
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

def count_original_data(data, key):
    st = set()
    for record in data:
        if record['verification_feedback'].lower() == 'incorrect':
            st.add(record[key])
    
    return len(st)

# def get_q_id(question):
#     for q in questions:
#         if question.lower() == questions[q]['question']:
#             return q

def get_questions(data):
    q_set = set()
    for record in data:
        q_set.add(record['question'])
    
    return q_set

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

    for _ in range(12):
        random.shuffle(formatted)

    print(f'count: {count_data(formatted)}')
    return lower_case_data(formatted)

def lower_case_data(data):
    for record in data:
        for key, value in record.items():
            if isinstance(value, str):
                record[key] = value.lower()
    return data

# def generate_id(data):
#     for i, record in enumerate(data):
#         record['id'] = f'smp{i:04d}{get_q_id(record['question'])}'
    
#     return data

def curate_data():
    questions = list(get_json('data/metadata/acceptable.json').keys())

    for i in ['unseen_answers.json']:
        curated = []
        rejected = []
        data = get_json(f'data/updated/formatted/{i}')
        for record in data:
            if record['id'][-4:] in questions:
                curated.append(record)
            else:
                rejected.append(record)

        print(f'{i}: Rejected: {len(rejected)}')
        save_json(rejected, f'./data/rejected/{i}')

        print(f'{i}: Accepted: {len(curated)}')
        save_json(curated, f'./data/curated/{i}')

if __name__ == '__main__':
    data = get_json('data/updated/formatted/unseen_questions.json')
    data = separate(data)

    for i in data:
        print(f'{i}: {len(data[i])}')
    
else:
    print(__name__, '\n\n')