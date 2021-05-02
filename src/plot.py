import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
from simple_classifiers import generate_ngram_values, generate_tf_idf_values
import pickle


def predict_(text, obj):
    counts = obj[0].transform(text.astype('U'))
    tfidf = obj[1].transform(counts.astype('U'))
    return tfidf



def plot_confusion_mtx(cm, labels, title, model):
    cmap=plt.cm.Blues

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    plt.xticks(marks, labels, rotation=45)
    plt.yticks(marks, labels)
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(f"../results/plots/confusion_matrix_{model}.png")
    plt.show()



def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest, title):  # Do it separately
    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'None',
                                        1: 'Rasism/sexism',
                                        2: 'Hate speech',
                                        3: 'Offensive language',
                                        4: 'Profanity',
                                        5: 'Islamophobia'})

    # Fitting to the training data first then scoring with the test data
    visualizer.fit(xtrain, ytrain)
    visualizer.score(X=xtest, y=ytest)
    visualizer.show(outpath=f"../results/plots/roc_curve_{title}.png")



def plot_datasets():
    df = pd.read_csv('../data/processed/combined_data.csv')
    classes = []
    for l in df['label']:
        if l == 0:
            classes.append('None')
        elif l == 1:
            classes.append('Rasism/sexism')
        elif l == 2:
            classes.append('Hate Speech')
        elif l == 3:
            classes.append('Offensive language')
        elif l == 4:
            classes.append('Profanity')
        elif l == 5:
            classes.append('Islamophobia')
    df['labels'] = classes
    df["labels"].value_counts().plot(kind="bar", title=f"Class distribution of combined dataset in english")
    plt.savefig('../results/plots/combined_dataset_distribution.png')
    plt.show()



if __name__ == '__main__':
    plot_datasets()
    df = pd.read_csv('../results/train_data.csv')
    x_train = df['x_train']
    y_train = df['y_train']
    df = pd.read_csv('../results/test_data.csv')
    x_test = df['x_test']
    y_test = df['y_test']

    X_train_counts, count_vec = generate_ngram_values(x_train)
    X_train_tdidf, tf_transformer = generate_tf_idf_values(X_train_counts)

    with open('../models/features/count_vector.pkl', 'rb') as file:
        count_vec = pickle.load(file)
    with open('../models/features/tfidf_transformer.pkl', 'rb') as file:
        tfidf_trans = pickle.load(file)
    with open('../models/logistic_regression_model.pkl', 'rb') as file:
        clf_lr = pickle.load(file)

    with open('../models/svm_model.pkl', 'rb') as file:
        clf_svm = pickle.load(file)

    with open('../models/multinomial_nb_model.pkl', 'rb') as file:
        clf_nb = pickle.load(file)

    y_pred_lr = predict_(x_test, [count_vec, tfidf_trans, clf_lr])
    y_pred_svm = predict_(x_test, [count_vec, tfidf_trans, clf_svm])
    y_pred_nb = predict_(x_test, [count_vec, tfidf_trans, clf_nb])

    print("Logistic Regression evaluation ...")
    plot_ROC_curve(LogisticRegression(C=100, class_weight='balanced', solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr'), X_train_tdidf, y_train, y_pred_lr, y_test, 'lr')
    print("SVM evaluation ...")
    plot_ROC_curve(SVC(), X_train_tdidf, y_train, y_pred_svm, y_test, 'svm')
    print("NB evaluation ...")
    plot_ROC_curve(MultinomialNB(), X_train_tdidf, y_train, y_pred_nb, y_test, 'nb')
    print("Completed.")
