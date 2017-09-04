import word2vec
from read_location_names import read_location_names
import re
import jieba

embedding_model_path = './embedding_files/embedding.bin'
word_cluster_path = './embedding_files/word-clusters.txt'
words_path = './embedding_files/words.txt'
sentences_path = './embedding_files/sentences.txt'
non_location_sample_path = './embedding_files/non_location_samples.txt'

embedding_dim = 50

def word_segmentation():
    s = open(sentences_path, 'r', encoding='utf-8')
    w = open(words_path, 'w', encoding='utf-8')
    # non_location_samples = open(non_location_sample_path, 'w', encoding='utf-8')
    # locations, locations_with_suffix = read_location_names()
    # location_pattern = '(' + ')|('.join(locations_with_suffix + locations) + ')'
    # location_set = set(locations_with_suffix + locations)
    skip_next_line = True
    # met_next = True
    for line in s:
        if skip_next_line:
            skip_next_line = False
            continue
        try:
            sentence = line.split()[0]
        except Exception:
            # if met_next:
            #     w.write('\n')
            #     met_next = False
            skip_next_line = True
            continue
        # met_next = True
        words = jieba.cut(sentence, cut_all=False)
        w.write(' '.join(words))
        # sentence = re.sub(location_pattern, '', sentence)
        # words = list(filter(lambda w: w in location_set, words))
        # non_location_samples.write(' '.join(words))
        w.write('\n')

def generate_non_location_samples():
    w = open(words_path, 'r', encoding='utf-8')
    non_location_samples = []
    locations, locations_with_suffix = read_location_names()
    location_set = set(locations_with_suffix + locations)
    for line in w:
        non_location_samples += filter(lambda w: w not in location_set, line.split())

    return non_location_samples


def train_embedding():
    word2vec.word2phrase(words_path, './embedding_files/phrases', verbose=True)
    word2vec.word2vec('./embedding_files/phrases', embedding_model_path, size=embedding_dim, verbose=True)
    word2vec.word2clusters('./embedding_files/words.txt', word_cluster_path, 100, verbose=True)

# word_segmentation()
# generate_non_location_samples()
# train_embedding()
