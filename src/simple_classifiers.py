import pickle

import pandas as pd
from preprocess_data import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from utils import *


def save_model(clf, modelPath):
    with open(modelPath, 'wb') as model:
        pickle.dump(clf, model, pickle.HIGHEST_PROTOCOL)


def save_features(count_vec, tfidf, vecPath, tfidfPath):
    with open(vecPath, 'wb') as final_count_vec:
        pickle.dump(count_vec, final_count_vec, pickle.HIGHEST_PROTOCOL)
    with open(tfidfPath, 'wb') as final_tf_transformer:
        pickle.dump(tfidf, final_tf_transformer, pickle.HIGHEST_PROTOCOL)


def get_logistic_regression_model(x_train, y_train):
    print("Training Logistic Regression model...")
    lr = LogisticRegression(C=100, class_weight='balanced', solver='liblinear',
                            penalty='l2', max_iter=100, multi_class='ovr')
    lr_clf = lr.fit(x_train, y_train)
    print("Logistic Regression model successfully trained.")
    savePath = '../models/logistic_regression_model.pkl'
    save_model(lr_clf, savePath)


def get_gaussian_nb_model(x_train, y_train):  # Not working with sparse csr_matrix
    print("Training Gaussian Naive Bayes model...")
    gnb = GaussianNB()

    # x_train = np.array(x_train)
    gnb_clf = gnb.fit(x_train, y_train)
    print("Gaussian Naive Bayes model successfully trained.")
    savePath = '../models/gaussian_nb_model.pkl'
    save_model(gnb_clf, savePath)


def get_multinomial_nb_model(x_train, y_train):
    print("Training Multinomial Naive Bayes model...")
    gnb = MultinomialNB()

    # x_train = np.array(x_train)
    gnb_clf = gnb.fit(x_train, y_train)
    print("Multinomial Naive Bayes model successfully trained.")
    savePath = '../models/multinomial_nb_model.pkl'
    save_model(gnb_clf, savePath)


def get_svm_model(x_train, y_train):
    print("Training SVM model...")
    svm = SVC()
    svm_clf = svm.fit(x_train, y_train)
    print("SVM model successfully trained.")
    savePath = '../models/svm_model.pkl'
    save_model(svm_clf, savePath)


def generate_ngram_values(x_train):
    count_vect = CountVectorizer(ngram_range=(1, 3), stop_words='english',
                                preprocessor=preprocess, lowercase=True)
    X_train_counts = count_vect.fit_transform(x_train.values.astype('U'))
    return [X_train_counts, count_vect]


def generate_tf_idf_values(x_counts):
    tf_transformer = TfidfTransformer(norm='l2', use_idf=True,
                                      smooth_idf=True, sublinear_tf=False)
    X_train_tfidf = tf_transformer.fit_transform(x_counts)
    return [X_train_tfidf, tf_transformer]


if __name__ == '__main__':
    data = pd.read_csv(DATASET, index_col=False)
    labels, tweets = data['label'], data['text']
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33, random_state=42)

    train_data = pd.DataFrame()
    train_data['x_train'] = X_train
    train_data['y_train'] = y_train
    train_data.to_csv('../results/train_data.csv')


    test_data = pd.DataFrame()
    test_data['x_test'] = X_test
    test_data['y_test'] = y_test
    test_data.to_csv('../results/test_data.csv')

    X_train_counts, count_vec = generate_ngram_values(X_train)
    X_train_tdidf, tf_transformer = generate_tf_idf_values(X_train_counts)

    print('Train set length, X: ', len(X_train))
    print('Train set length, y:', len(y_train))
    print('Test set length, X:', len(X_test))
    print('Test set length, y:', len(y_test))


    save_features(count_vec, tf_transformer,
                  '../models/features/count_vector.pkl',
                  '../models/features/tfidf_transformer.pkl')

    get_logistic_regression_model(X_train_tdidf, y_train)
    get_svm_model(X_train_tdidf, y_train)
    get_multinomial_nb_model(X_train_tdidf, y_train)






