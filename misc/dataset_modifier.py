import json
import random
import re

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

def save_json(data, path, ask=True):
    inp = input('Do you want to save? [Y/n]: ').lower() if ask == True else 'y'
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
    
def find_data(data, id):
    for record in data:
        if record is None: continue
        if record['id'] == id:
            return record
    
    else:
        return None
    
def extract_prediction(text, ID):
        if not isinstance(text, str):
            return None

        # Find JSON object that contains "predicted_label"
        match = re.search(
            r'\{\s*"predicted_label"\s*:\s*"(correct|partial|incorrect|uncertain)"\s*,\s*"confidence"\s*:\s*[0-9.]+\s*\}',
            text,
            re.S
        )

        if not match:
            return None

        try:
            data = json.loads(match.group())
            return {
                "ID": ID,
                "predicted_label": data["predicted_label"],
                "confidence": float(data["confidence"])
            }
        except Exception:
            return None

def merge_audit(auditted_result, aug):
    audited_s = []
    audited_u = []
    discarded = []
    for record in aug:

        audit = find_data(auditted_result, record['id'])
        if audit == None:
            audited_u.append(record)

        else:
            print(audit)
            if audit['confidence'] <= .5:
                record['audit'] = audit['predicted_label']
                record['confidence'] = audit['confidence']
                audited_u.append(record)
            elif audit['predicted_label'] == record['verification_feedback']:
                record['audit'] = audit['predicted_label']
                record['confidence'] = audit['confidence']
                audited_s.append(record)
            else:
                record['confidence'] = audit['confidence']
                record['audit'] = audit['predicted_label']
                discarded.append(record)
    print('passed: ', len(audited_s))
    save_json(audited_s, './data/augmented/passed.json')
    print('uncertain: ', len(audited_u))
    save_json(audited_u, './data/augmented/uncertain.json')
    print('discarded: ', len(discarded))
    save_json(discarded, './data/augmented/discarded.json')

if __name__ == '__main__':
    augmented_set = get_json('./data/augmented/passed.json')
    original_set = get_json('./data/curated/train.json')

    audits = get_json('./data/metadata/audit.json')

    # merge_audit(audits, augmented_set)

    # for record in original_set:
    #     pass
    # data = combine_data(augmented_set, original_set)
    # for _ in range(3):
    #     random.shuffle(data)
    # save_json(data, './data/updated/combined_set/train.json')

    
    
else:
    print(__name__, '\n\n')