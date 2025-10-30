import json

files = ['train', 'validation', 'unseen_questions', 'unseen_answers']
output = []
for fp in files:
    p_count = 0
    c_count = 0
    i_count = 0
    with open(f"./data/{fp}.json", 'r', encoding='utf8') as f:
        training_data = json.load(f)
        f.close()
    for record in training_data:
        if record['verification_feedback'].lower() == 'correct':
            c_count += 1
        if record['verification_feedback'].lower() == 'partially correct':
            p_count += 1
        if record['verification_feedback'].lower() == 'incorrect':
            i_count += 1

    st = f'{fp}:\n===================================\nCorrect: {c_count}\nPartially Correct: {p_count}\nIncorrect: {i_count}\nTotal: {c_count+i_count+p_count}\n-----------------------------------' 
    print(st)
    output.append(st)

with open('./data/info.txt', 'w') as f:
    f.writelines(i+'\n\n' for i in output)
        
# scores = [0.5, 1.0, 1.5, 2.5, 3.5]
        
# print(f'{record['id']}-> Score: {record['score']}, Maximum Score: {record['max_score']}')

# if record['verification_feedback'].lower() == 'correct':
#     record['max_score'] = record['score']