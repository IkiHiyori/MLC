import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import time


def get_data(data_path):
    data = []
    labels = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            data_dic = json.loads(line)  # keys=['id', 'title', 'abstract', 'section', 'subsection', 'group', 'labels']
            str = ' '
            data.append(str.join(data_dic['abstract']))
            labels.append(data_dic['labels'])
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '))
    tf_idf_csr = tfidfVecorizer.fit_transform(data)  # csr matrix
    # labels to csr matrix
    row = []
    col = []
    for (i, label) in enumerate(labels):
        for count in range(len(label)):
            row.append(i)
        for j in label:
            col.append(j)
    ones = [1] * len(row)
    labels_csr = csr_matrix((ones, (row, col)), shape=(len(data), 798))
    return tf_idf_csr, labels_csr
