import word2vec
import jieba

embedding_model_path = './embedding_files/embedding_files.bin'
word_cluster_path = './embedding_files/word-clusters.txt'
embedding_dim = 50

def word_segmentation():
    s = open('./embedding_files/sentences.txt', 'r', encoding='utf-8')
    w = open('./embedding_files/words.txt', 'w', encoding='utf-8')
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
        w.write(' '.join(jieba.cut(sentence, cut_all=False)))
        w.write('\n')

def train_embedding():
    word2vec.word2phrase('./embedding_files/words.txt', './embedding_files/phrases', verbose=True)
    word2vec.word2vec('./embedding_files/phrases', embedding_model_path, size=embedding_dim, verbose=True)
    word2vec.word2clusters('./embedding_files/words.txt', word_cluster_path, 100, verbose=True)

# word_segmentation()
train_embedding()
