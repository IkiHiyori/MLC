import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import time


def get_data_csr(train_path, test_path):
    train, train_labels = read_data(train_path)
    train_labels_csr = labels_to_csr(train_labels, csr_shape=(len(train), 798))  # 0~797
    test, test_labels = read_data(test_path)
    test_labels_csr = labels_to_csr(test_labels, csr_shape=(len(test), 798))
    tfidf_train = TfidfVectorizer(analyzer=lambda x: x.split(' '))
    train_csr = tfidf_train.fit_transform(train)  # csr matrix
    tfidf_test = TfidfVectorizer(vocabulary=tfidf_train.vocabulary_)
    test_csr = tfidf_test.fit_transform(test)
    return train_csr, train_labels_csr, test_csr, test_labels_csr


def read_data(path):
    data = []
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data_dic = json.loads(line)  # keys=['id', 'title', 'abstract', 'section', 'subsection', 'group', 'labels']
            str = ' '
            data.append(str.join(data_dic['abstract']))
            labels.append(data_dic['labels'])
    return data, labels


def labels_to_csr(raw_labels, csr_shape):
    # labels to csr matrix
    row = []
    col = []
    for (i, label) in enumerate(raw_labels):
        for count in range(len(label)):
            row.append(i)
        for j in label:
            col.append(j)
    ones = [1] * len(row)
    labels_csr = csr_matrix((ones, (row, col)), shape=csr_shape)
    return labels_csr


if __name__ == '__main__':
    start = time.clock()
    train_path = 'dataset/train.json'
    test_path = 'dataset/test.json'
    train_csr, train_labels_csr, test_csr, test_labels_csr = get_data_csr(train_path, test_path)
    end = time.clock()
    print('time: {: .2f} s'.format(end - start))

