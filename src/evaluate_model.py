import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import *
from plot import *


def predict_(text, obj):
    # data = []
    # for t in text:
    #     print(t.)
    #     data.append(t)
    counts = obj[0].transform(text.astype('U'))
    tfidf = obj[1].transform(counts.astype('U'))
    prediction = obj[2].predict(tfidf)
    return prediction


def evaluate(x_test, y_test, model, grouped_labels):
    with open('../models/features/count_vector.pkl', 'rb') as file:
        count_vec = pickle.load(file)
    with open('../models/features/tfidf_transformer.pkl', 'rb') as file:
        tfidf_trans = pickle.load(file)
    model_type = ''
    if model=='lr':
        model_type = 'logistic_regression_model.pkl'
    elif model=='svm':
        model_type = 'svm_model.pkl'
    elif model=='nb':
        model_type = 'multinomial_nb_model.pkl'
    with open('../models/' + model_type, 'rb') as file:
        clf = pickle.load(file)

    y_pred = predict_(x_test, [count_vec, tfidf_trans, clf])

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='micro')
    f1_score = metrics.f1_score(y_test, y_pred, average='micro')
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # roc_auc = metrics.auc(fpr, tpr)
    class_report = metrics.classification_report(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm = (cm.T / grouped_labels).T
    # fp = cm.sum(axis=0) - np.diag(cm)
    # fn = cm.sum(axis=1) - np.diag(cm)
    # tp = np.diag(cm)
    # tn = cm.values.sum() - (fp + fn + tp)
    # fpr = fp / (fp +

    # plot_roc_curve(model, y_pred, y_test)


    return accuracy, precision, f1_score, cm, class_report


def evaluate_lr(x_test, y_test, grouped_labels):
    accuracy, precision, f1_score, cm, class_report = evaluate(x_test, y_test, 'lr', grouped_labels)
    title = "Logistic Regression confusion matrix"
    plot_confusion_mtx(cm, ['None', 'Rasism/sexism', 'Hate Speech', 'Offensive language', 'Profanity', 'Islamophobia'], title, 'lr')
    # plot_roc_curve(fpr, tpr, roc_auc, "lr")
    return accuracy, precision, f1_score


def evaluate_svm(x_test, y_test, grouped_labels):
    accuracy, precision, f1_score, cm, class_report = evaluate(x_test, y_test, 'svm', grouped_labels)
    title = "SVM confusion matrix"
    plot_confusion_mtx(cm, ['None', 'Rasism/sexism', 'Hate Speech', 'Offensive language', 'Profanity',
                                    'Islamophobia'], title, 'svm')
    # plot_roc_curve("SVM", y_pred, y_test)
    return accuracy, precision, f1_score

def evaluate_nb(x_test, y_test, grouped_labels):
    accuracy, precision, f1_score, cm, class_report = evaluate(x_test, y_test, 'nb', grouped_labels)
    title = "Multinomial Naive Bayes confusion matrix"
    plot_confusion_mtx(cm, ['None', 'Rasism/sexism', 'Hate Speech', 'Offensive language', 'Profanity',
                                    'Islamophobia'], title, 'nb')
    # plot_roc_curve(fpr, tpr, roc_auc, "nb")
    return accuracy, precision, f1_score


if __name__ == '__main__':
    data = pd.read_csv(TEST_DATASET, index_col=False)
    x_test = data['x_test']
    y_test = data['y_test']
    grouped_classes = data.groupby('y_test').size()
    grouped_classes = np.array(list(grouped_classes))

    print("Evaluating logistic regression...")
    lr_accuracy, lr_precision, lr_f1_score = evaluate_lr(x_test, y_test, grouped_classes)
    print("Evaluation SVM ...")
    svm_accuracy, svm_precision, svm_f1_score = evaluate_svm(x_test, y_test, grouped_classes)
    print("Evaluation Naive Bayes ...")
    nb_accuracy, nb_precision, nb_f1_score = evaluate_nb(x_test, y_test, grouped_classes)
    print("Evaluated successfully.")
    df = pd.DataFrame()
    df['Classifier'] = ['LogisticRegression', 'SVM', 'MultinomialNB']
    df['Accuracy'] = [str(lr_accuracy), str(svm_accuracy), str(nb_accuracy)]
    df['Precision'] = [str(lr_precision), str(svm_precision), str(nb_precision)]
    df['F1-score'] = [str(lr_f1_score), str(svm_f1_score), str(nb_f1_score)]

    df.to_csv('../results/metrics.csv')
