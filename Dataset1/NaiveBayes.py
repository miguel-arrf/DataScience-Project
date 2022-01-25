from pandas import unique
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class NaiveBayes:
    def __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        estimators = {'GaussianNB': GaussianNB(),
                      'BernoulliNB': BernoulliNB()
                      }

        xvalues_accuracy = []
        xvalues_precision = []
        xvalues_recall = []

        yvalues_accuracy = []
        yvalues_precision = []
        yvalues_recall = []

        scores = [(precision_score, yvalues_precision, xvalues_precision), (recall_score, yvalues_recall, xvalues_recall)]

        for score in scores:
            for clf in estimators:
                score[2].append(clf)
                estimators[clf].fit(trnX, trnY)
                prdY = estimators[clf].predict(tstX)
                print(confusion_matrix(tstY, prdY))
                score[1].append(score[0](tstY, prdY, pos_label="Killed"))

        for clf in estimators:
            xvalues_accuracy.append(clf)
            estimators[clf].fit(trnX, trnY)
            prdY = estimators[clf].predict(tstX)
            yvalues_accuracy.append(accuracy_score(tstY, prdY))

        figure()
        bar_chart(xvalues_accuracy, yvalues_accuracy, title='Comparison of Naive Bayes Models', ylabel='accuracy',
                  percentage=True)
        savefig(f'images/nb/nb_study_accuracy.png')
        show()

        figure()
        bar_chart(xvalues_precision, yvalues_precision, title='Comparison of Naive Bayes Models', ylabel='precision',
                  percentage=True)
        savefig(f'images/nb/nb_study_precision.png')
        show()

        figure()
        bar_chart(xvalues_recall, yvalues_recall, title='Comparison of Naive Bayes Models', ylabel='recall',
                  percentage=True)
        savefig(f'images/nb/nb_study_recall.png')
        show()

