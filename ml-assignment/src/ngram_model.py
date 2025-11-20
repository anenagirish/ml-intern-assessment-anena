import random
import re
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        self.trigrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.trained = False

    def clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return text.split()

    def fit(self, text):
        # Handle empty or invalid text
        if not text or text.strip() == "":
            self.trained = False
            return

        tokens = self.clean_and_tokenize(text)

        if len(tokens) < 2:
            self.trained = False
            return

        # Add padding tokens
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]
        self.vocab = set(tokens)

        # Count trigrams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            self.trigrams[(w1, w2)][w3] += 1

        self.trained = True

    def generate(self, max_length=50):
        # If model is not trained, return empty string
        if not self.trained:
            return ""

        w1, w2 = "<s>", "<s>"
        output = []

        for _ in range(max_length):
            next_word_dict = self.trigrams.get((w1, w2), None)

            if not next_word_dict:
                break

            # Choose next word probabilistically
            words = list(next_word_dict.keys())
            counts = list(next_word_dict.values())
            total = sum(counts)
            probabilities = [c / total for c in counts]

            w3 = random.choices(words, probabilities)[0]

            if w3 == "</s>":
                break

            output.append(w3)
            w1, w2 = w2, w3

        return " ".join(output)
