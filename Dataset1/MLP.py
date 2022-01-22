from matplotlib import pyplot as plt
from matplotlib.pyplot import show, figure, savefig, subplots
from pandas import unique
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier

from ds_charts import multiple_line_chart, plot_evaluation_results, HEIGHT


class MLP:
    def __init__(self, trnX, trnY, tstY, tstX):
        labels = unique(trnY)
        labels.sort()

        lr_type = ['constant', 'invscaling', 'adaptive']
        max_iter = [100, 500, 1000, 2500]
        learning_rate = [0.01, .1, .5]
        best = ('', 0, 0)
        last_best = 0
        best_model = None

        cols = len(lr_type)
        figure()
        fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        for k in range(len(lr_type)):
            print("k: ", lr_type[k])

            d = lr_type[k]
            values = {}
            for lr in learning_rate:
                print("\tlr: ", lr)

                yvalues = []
                for n in max_iter:
                    print("\t\tn: ", n)

                    mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                        learning_rate_init=lr, max_iter=n, verbose=False)
                    mlp.fit(trnX, trnY)
                    prdY = mlp.predict(tstX)
                    yvalues.append(precision_score(tstY, prdY, pos_label="Killed"))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_model = mlp
                values[lr] = yvalues
            multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                                xlabel='mx iter', ylabel='precision', percentage=True)
        savefig(f'images/mlp/mlp_study.png')
        #show()
        print(
            f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with precision={last_best}')

        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst, extra="")
        savefig(f'images/mlp/mlp_best.png')
        #show()

        # Overfitting
        def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel, extra=""):
            evals = {'Train': prd_trn, 'Test': prd_tst}
            plt.figure()
            multiple_line_chart(xvalues, evals, ax=None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel,
                                percentage=True)
            plt.savefig(f'images/mlp/overfitting_{extra}.png')

        lr_type = best[0]
        lr = best[1]
        y_tst_values = []
        y_trn_values = []
        for n in max_iter:
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr,
                                max_iter=n, verbose=False)
            mlp.fit(trnX, trnY)
            prd_tst_Y = mlp.predict(tstX)
            prd_trn_Y = mlp.predict(trnX)
            y_tst_values.append(precision_score(tstY, prd_tst_Y, pos_label="Killed"))
            y_trn_values.append(precision_score(trnY, prd_trn_Y, pos_label="Killed"))
        plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}',
                               xlabel='nr episodes', ylabel="precision")

        f = open(f"images/mlp/bestModel.txt", "w")
        f.write(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with precision={last_best}')
        f.close()
