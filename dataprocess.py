import json
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def get_data_matrix(data_path):
    data = []
    labels = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            data_dic = json.loads(line)  # keys=['id', 'title', 'abstract', 'section', 'subsection', 'group', 'labels']
            str = ' '
            data.append(str.join(data_dic['abstract']))
            labels.append(data_dic['labels'])
    return data, labels


def get_tf_idf_matrix(data):
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '))
    tf_idf_matrix = tfidfVecorizer.fit_transform(data)   # csr matrix
    word_id_dic = tfidfVecorizer.vocabulary_
    return tf_idf_matrix, word_id_dic

if __name__ == '__main__':
    start = time.clock()
    data, labels = get_data_matrix('dataset/test.json')
    tf_idf_matrix, word_id_dic = get_tf_idf_matrix(data)
    end = time.clock()
    print('time:', end - start)
