from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self, file_tag, filename, target, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        estimators = {'GaussianNB': GaussianNB(),
                      'MultinomialNB': MultinomialNB(),
                      'BernoulliNB': BernoulliNB()
                      }

        xvalues = []
        yvalues = []
        for clf in estimators:
            xvalues.append(clf)
            estimators[clf].fit(trnX, trnY)
            prdY = estimators[clf].predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))

        figure()
        bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        savefig(f'images/{file_tag}_nb_study.png')
        show()
