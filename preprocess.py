import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
MAX_LEN = 300


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        texts, scores = [], []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                texts.append(parts[2])
                scores.append(float(parts[1]))
    return texts, np.array(scores)

def load_fold_data(fold_path):
    train_texts, train_scores = read_tsv(os.path.join(fold_path, "train.tsv"))
    dev_texts, dev_scores = read_tsv(os.path.join(fold_path, "dev.tsv"))
    test_texts, test_scores = read_tsv(os.path.join(fold_path, "test.tsv"))
    return (train_texts, train_scores), (dev_texts, dev_scores), (test_texts, test_scores)

def compute_vocab_richness(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    unique_words = set(words)
    richness = len(unique_words) / (len(words) + 1e-5)
    return richness

def compute_readability(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    if len(sentences) == 0 or len(words) == 0:
        return 0
    words_per_sentence = len(words) / len(sentences)
    return words_per_sentence  # Simplified readability metric

def compute_coherence(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    overlaps = []
    for i in range(1, len(sentences)):
        prev_words = set(word_tokenize(sentences[i - 1].lower())) - stop_words
        curr_words = set(word_tokenize(sentences[i].lower())) - stop_words
        common = prev_words.intersection(curr_words)
        overlaps.append(len(common))
    return sum(overlaps) / (len(overlaps) + 1e-5) if overlaps else 0

def extract_features(text):
    return np.array([
        compute_vocab_richness(text),
        compute_readability(text),
        compute_coherence(text)
    ])
