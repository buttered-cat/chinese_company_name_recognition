import gensim
from read_location_names import read_location_names
import re
import jieba

# embedding_model_path = './embedding_files/embedding.bin'
embedding_model_path = './embedding_files/embedding_model'
# word_cluster_path = './embedding_files/word-clusters.txt'
segmented_words_path = './data/segmented_words.txt'
sentences_path = './data/raw_sentences.txt'
# non_location_sample_path = './embedding_files/non_location_samples.txt'
augmented_corpus_path = './data/augmented_corpus.txt'

embedding_dim = 20

def word_segmentation():
    raw_sentences = open(sentences_path, 'r', encoding='utf-8')
    segmented_words = open(segmented_words_path, 'w', encoding='utf-8')
    augemented_data = open(augmented_corpus_path, 'w', encoding='utf-8')

    locations, locations_with_suffix = read_location_names()
    # segment locations
    location_list = [jieba.cut(l, cut_all=False) for l in locations_with_suffix + locations]

    skip_next_line = True
    # met_next = True
    for line in raw_sentences:
        if skip_next_line:
            skip_next_line = False
            continue
        try:
            sentence = line.split()[0]
        except Exception:
            # blank line, skip next line which is company name
            skip_next_line = True
            continue
        words = jieba.cut(sentence, cut_all=False)

        # write segmented words
        segmented_words.write(' '.join(words))
        segmented_words.write('\n')

        # generate augmented data
        augmented_corpus = generate_augmented_corpus_sentence(words, location_list)
        for s in augmented_corpus:
            # write segmented words
            augemented_data.write(' '.join(s))
            augemented_data.write('\n')

def generate_augmented_corpus_sentence(word_list, location_list):
    # , keep_sentence_structure=True):
    # w = open(segmented_words_path, 'r', encoding='utf-8')
    # generated_samples = []

    location_set = set(location_list)
    generated_corpus = []

    # TODO: there's some inconsistency, here it assumes locations consist of a single word
    for l in location_list:
        generated_corpus.append([w for c in word_list for w in ([c] if c not in location_set else l)])
    return generated_corpus

    # resplit = False if re.search(r' ', location) is None else True
    # empty_str = True if location == '' else False
    #
    # def map_cond(w):
    #     if w in location_set:
    #         return location
    #     else:
    #         return w
    #
    # if resplit:
    #     line = ' '.join(map(map_cond, word_list)).split()
    # else:
    #     line = map(map_cond, word_list)
    # # if keep_sentence_structure:
    # #     generated_samples.append(filter(lambda w: w != '', line))
    # # else:
    # if empty_str:
    #     generated_samples = (filter(lambda w: w != '', line))
    # else:
    #     generated_samples = line
    #
    # return generated_samples


def generate_non_locations(location_set):
    segmented_words = open(segmented_words_path, 'r', encoding='utf-8')
    # flattened data
    words = []
    for line in segmented_words:
        words += [w for w in line.split() if w not in location_set]

    return words

def train_embedding():
    w = open(augmented_corpus_path, 'r', encoding='utf-8')
    sentences = []
    for line in w:
        sentences.append(line.split())
    embedding = gensim.models.Word2Vec(sentences, min_count=1, sg=1, size=embedding_dim, iter=5)

    embedding.save(embedding_model_path)

word_segmentation()
# generate_non_location_samples()
# train_embedding()
