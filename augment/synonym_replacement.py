import random
import spacy
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")


class SynonymAugmenter:
    def __init__(self, max_words=5, replace_n=3, similarity_lb=0.7, similarity_ub=0.85,
                 length_lb=0.7, length_ub=1.3, max_attempts=20):
        # Models
        self._nlp = spacy.load("en_core_web_sm")
        self._embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._stopwords = self._nlp.Defaults.stop_words
        self.max_words = max_words
        self.replace_n = replace_n
        self.similarity_lb = similarity_lb
        self.similarity_ub = similarity_ub
        self.length_lb = length_lb
        self.length_ub = length_ub
        self.max_attempts = max_attempts

    # -------- Helpers --------
    def _get_synonyms(self, word, pos_tag):
        pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
        if pos_tag not in pos_map:
            return []

        syns = set()
        for synset in wn.synsets(word, pos_map[pos_tag]):
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace("_", " ").lower()
                if lemma_name == word or not lemma_name.isalpha():
                    continue
                if len(lemma_name.split()) != 1 or abs(len(lemma_name) - len(word)) > 6:
                    continue
                syns.add(lemma_name)
        return list(syns)

    def _candidate_words(self, doc):
        candidates = []
        for token in doc:
            if token.is_stop or token.text.lower() in self._stopwords or token.ent_type_:
                continue
            if token.pos_ not in ["NOUN", "VERB", "ADJ", "ADV"] or not token.text.isalpha() or len(token.text) < 3:
                continue
            syns = self._get_synonyms(token.text.lower(), token.pos_)
            if syns:
                candidates.append((token.text, token.pos_, syns))
        return candidates

    def _replace_words(self, text, replacements):
        new_text = text
        for orig, repl in replacements.items():
            new_text = new_text.replace(orig, repl)
        return new_text

    def semantic_sim(self, a, b):
        em_a = self._embed.encode(a, convert_to_tensor=True)
        em_b = self._embed.encode(b, convert_to_tensor=True)
        return util.cos_sim(em_a, em_b).item()
    
    def deviation_score(sim, lb=0.7, ub=0.85):
        if lb <= sim <= ub:
            # In bounds → distance from upper bound (smaller is better)
            return ub - sim
        else:
            # Out of bounds → distance to nearest bound
            return min(abs(sim - lb), abs(sim - ub)) + 1 

    def valid_grammar(self, text):
        doc = self._nlp(text)
        if len(doc) == 0 or not doc.has_annotation("DEP"):
            return False
        return True

    # -------- Main augment function --------
    def augment(self, text):
        doc = self._nlp(text)
        candidates = self._candidate_words(doc)
        if len(candidates) < self.replace_n:
            return ""  # fallback to back-translation

        selected = random.sample(candidates, min(self.max_words, len(candidates)))

        attempts = 0
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                print("Max attempts exceeded. Returning fallback.")
                return ""

            replace_group = random.sample(selected, self.replace_n)
            repl_map = {orig: random.choice(syns) for orig, _, syns in replace_group}
            new_text = self._replace_words(text, repl_map)

            # QC: Grammar
            if not self.valid_grammar(new_text):
                continue

            # QC: Semantic similarity
            sim = self.semantic_sim(text, new_text)
            if sim < self.similarity_lb or sim > self.similarity_ub:
                continue

            # QC: Length
            if len(new_text) < len(text) * self.length_lb or len(new_text) > len(text) * self.length_ub:
                continue

            print(f"Augmentation succeeded after {attempts} attempts, similarity={sim:.3f}")
            return new_text
