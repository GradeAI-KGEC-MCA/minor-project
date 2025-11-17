# from misc.dataset_modifier import get_json, json
# from misc.synonym_replacement import synonym_augment

# data = get_json('./data/train.json')
# res = []

# for record in data:
#     text = record['provided_answer']
#     aug_samples = synonym_augment(text)

#     for t in aug_samples:
#         aug_record = record.copy()
#         aug_record['provided_answer'] = t
#         res.append(aug_record)

# path = './data/augmented/synonym_replacement.json'
# with open(path, 'w', encoding='utf8') as f:
#     json.dump(res, f, indent=4)

import misc.back_translation
