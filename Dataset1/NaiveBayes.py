from matplotlib import pyplot as plt
from pandas import unique
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, plot_evaluation_results, HEIGHT, multiple_bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class NaiveBayes:
    def __init__(self, trnX, trnY, tstY, tstX, imagePath):
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

        scores = [(precision_score, yvalues_precision, xvalues_precision),
                  (recall_score, yvalues_recall, xvalues_recall)]

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

        fig, axs = plt.subplots(1, 1, figsize=(2 * HEIGHT, HEIGHT))
        evaluation = {
            'Accuracy': yvalues_accuracy,
            'Recall': yvalues_recall,
            'Precision': yvalues_precision
        }
        multiple_bar_chart(['GaussianNB', 'BernoulliNB'], evaluation, ax=axs, title=f"GaussianNB x BernoulliNB",
                           percentage=True)
        plt.tight_layout()
        savefig(f'{imagePath}/nb/nb_comparison.png', dpi=300)
        show()

        figure()
        bar_chart(xvalues_accuracy, yvalues_accuracy, title='Comparison of Naive Bayes Models', ylabel='accuracy',
                  percentage=True)
        savefig(f'{imagePath}/nb/nb_study_accuracy.png', dpi=300)
        show()

        figure()
        bar_chart(xvalues_precision, yvalues_precision, title='Comparison of Naive Bayes Models', ylabel='precision',
                  percentage=True)
        savefig(f'{imagePath}/nb/nb_study_precision.png', dpi=300)
        show()

        figure()
        bar_chart(xvalues_recall, yvalues_recall, title='Comparison of Naive Bayes Models', ylabel='recall',
                  percentage=True)
        savefig(f'{imagePath}/nb/nb_study_recall.png', dpi=300)
        show()

        clf = GaussianNB()
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)

        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra=f"Naive Bayes")
        savefig(f'{imagePath}/nb/nb_matrix.png', dpi=300)
        show()
