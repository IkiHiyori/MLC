import data_process
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.externals import joblib
import time


start = time.clock()

# data preparation
train_path = 'dataset/test.json'
test_path = 'dataset/validation.json'
train_csr, train_labels_csr, test_csr, test_labels_csr = data_process.get_data_csr(train_path, test_path)

# model definition
svm_model = OneVsRestClassifier(SVC(kernel='linear'))

# train
svm_model.fit(train_csr, train_labels_csr)
print('train time: {:5.4f} mins'.format((time.clock() - start)/60))

# save model
joblib.dump(svm_model, 'model/svm_model.m')
# svm_model = joblib.load('model/svm_model.m')

# test
predictions = svm_model.predict(test_csr)
# print(predictions)  # csr matrix

# metrics
precision = precision_score(test_labels_csr, predictions, average='micro')
recall = recall_score(test_labels_csr, predictions, average='micro')
f1 = f1_score(test_labels_csr, predictions, average='micro')
print('precision: {:5.4f} recall: {:5.4f} f1: {:5.4f}'.format(precision, recall, f1))

end = time.clock()
print('total time: ', (end - start)/60, 'mins')
