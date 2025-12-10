import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
import random
from nltk.corpus import wordnet as wn
import spacy
from sentence_transformers import SentenceTransformer, util
# load spaCy + SBERT
try:
    _nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")

_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

STOPWORDS = _nlp.Defaults.stop_words


# ---------- HELPERS ----------

def _get_synonyms(word, pos_tag):
    """POS-matched synonym extraction with heavy filtering."""
    pos_map = {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV
    }

    if pos_tag not in pos_map:
        return []

    syns = set()
    for synset in wn.synsets(word, pos_map[pos_tag]):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ").lower()

            # filters
            if lemma_name == word:
                continue
            if len(lemma_name.split()) != 1:
                continue
            if not lemma_name.isalpha():
                continue
            if abs(len(lemma_name) - len(word)) > 6:
                continue

            syns.add(lemma_name)

    return list(syns)


def _candidate_words(doc):
    """Extract good synonym-replaceable tokens."""
    candidates = []

    for token in doc:
        if token.is_stop:
            continue
        if token.text.lower() in STOPWORDS:
            continue
        if token.ent_type_:
            continue
        if token.pos_ not in ["NOUN", "VERB", "ADJ", "ADV"]:
            continue
        if not token.text.isalpha():
            continue
        if len(token.text) < 3:
            continue

        syns = _get_synonyms(token.text.lower(), token.pos_)
        if syns:
            candidates.append((token.text, token.pos_, syns))

    return candidates


def _replace_words(text, replacements):
    """Apply replacements into the raw text safely."""
    new_text = text
    for orig, repl in replacements.items():
        new_text = new_text.replace(orig, repl)
    return new_text


def semantic_sim(a, b):
    em_a = _embed.encode(a, convert_to_tensor=True)
    em_b = _embed.encode(b, convert_to_tensor=True)
    return util.cos_sim(em_a, em_b).item()


def valid_grammar(text):
    doc = _nlp(text)
    if len(doc) == 0:
        return False
    if doc.has_annotation("DEP") is False:
        return False
    return True


# ---------- MAIN AUGMENT FUNCTION ----------

def synonym_augment(text, max_words=5, replace_n=3):
    """
    text: input string
    max_words: pick up to N candidate tokens
    replace_n: replace exactly N words per variant
    """

    doc = _nlp(text)
    candidates = _candidate_words(doc)

    if len(candidates) < replace_n:
        return []

    # choose up to max_words from candidate list
    selected = random.sample(candidates, min(max_words, len(candidates)))

    attempts = 0

    threshold = {
        'similarity_ub': .85,
        'similarity_lb': .6,
        'length_ub': 1.3,
        'length_lb': .7
    }

    print('Generating replacements...')
    while True:
        attempts += 1
        if attempts >= 20:
            print('Attempts Exceeded. Proceeding with back-translation: ')
            return ''

        # choose N random words
        replace_group = random.sample(selected, replace_n)

        # build replacement map
        repl_map = {}
        for orig_word, pos, syns in replace_group:
            repl_map[orig_word] = random.choice(syns)

        new_text = _replace_words(text, repl_map)
        
        # ---------- QC: GRAMMAR ----------
        if not valid_grammar(new_text):
            continue

        # ---------- QC: SIMILARITY ----------
        sim = semantic_sim(text, new_text)
        if sim < threshold["similarity_lb"] or sim > threshold["similarity_ub"]:
            continue

        # ---------- QC: LENGTH ----------
        if len(new_text) < len(text) * threshold["length_lb"]:
            continue
        if len(new_text) > len(text) * threshold["length_ub"]:
            continue
        

        print(f'Succeeded at attempt: {attempts}')
        print(sim)
        return new_text