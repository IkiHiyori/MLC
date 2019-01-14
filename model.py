import data_process
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.externals import joblib
import warnings
import time


# data preparation
train_path = 'dataset/train.json'
test_path = 'dataset/test.json'
train_csr, train_labels_csr, test_csr, test_labels_csr = data_process.get_data_csr(train_path, test_path)


def train(model_type):
    start = time.clock()
    if model_type == 'svm':
        model = OneVsRestClassifier(SVC(kernel='linear'))
    elif model_type == 'dt':
        model = OneVsRestClassifier(DecisionTreeClassifier())
    else:
        warnings.warn('Invalid parameter(model type), using dt model', UserWarning)
        model = OneVsRestClassifier(DecisionTreeClassifier())
    print('training   : ' + model_type + ' model')
    model.fit(train_csr, train_labels_csr)
    joblib.dump(model, 'model/' + model_type + '/model.m')
    end = time.clock()
    print('train time : {:5.4f} mins'.format((end - start) / 60))
    return model


def test(model):
    start = time.clock()
    # test
    predictions = model.predict(test_csr)
    # save
    # joblib.dump(predictions, 'model/pred_labels')
    # metrics
    precision = precision_score(test_labels_csr, predictions, average='micro')
    recall = recall_score(test_labels_csr, predictions, average='micro')
    f1 = f1_score(test_labels_csr, predictions, average='micro')
    roc_auc = roc_auc_score(test_labels_csr.toarray(), predictions.toarray(), average='micro')
    print('precision: {:5.4f} recall: {:5.4f} f1: {:5.4f} roc_auc: {:5.4f}'.format(precision, recall, f1, roc_auc))
    end = time.clock()
    print('test time : {:5.4f} mins'.format((end - start) / 60))

# save model
# joblib.dump(dt_model, 'model/dt_model.m')

if __name__ == '__main__':
    model_type = 'dt'
    model = train(model_type)
    # model = joblib.load('model/dt_model.m')
    test(model)
