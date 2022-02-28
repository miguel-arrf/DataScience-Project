import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, show, subplots, savefig
from ds_charts import dummify, plot_line, multiple_line_chart, get_variable_types
from mlxtend.frequent_patterns import apriori, association_rules

data = read_csv('../teste_to_use_encoded.csv')
numeric_vars = get_variable_types(data)['Numeric']
# data.pop('Unnamed: 0')

for var in numeric_vars:
    data.pop(var)
data["PERSON_SEX"].replace(('F', 'M'), (1, 0), inplace=True)

data.pop("PERSON_INJURY")
data = data.sample(10000)

MIN_SUP: float = 0.05
var_min_sup = [0.2, 0.15, 0.13, 0.1, 0.05]

print("varminsup: ", var_min_sup)

patterns: DataFrame = apriori(data, min_support=MIN_SUP, use_colnames=True, verbose=True, max_len=10)
print(len(patterns), 'patterns')
nr_patterns = []
for sup in var_min_sup:
    pat = patterns[patterns['support'] >= sup]
    nr_patterns.append(len(pat))

figure(figsize=(6, 4))
plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')
plt.tight_layout()

savefig("images/default/nr_patterns_support.png")
show()

MIN_CONF: float = 0.1
rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
print(f'\tfound {len(rules)} rules')


def plot_top_rules(rules: DataFrame, metric: str, per_metric: str) -> None:
    _, ax = subplots(figsize=(25, 3))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(f'TOP 10 per Min {per_metric} - {metric}', fontweight="bold")
    text = ''
    cols = ['antecedents', 'consequents']
    rules[cols] = rules[cols].applymap(lambda x: tuple(x))
    for i in range(len(rules)):
        rule = rules.iloc[i]
        text += f"{rule['antecedents']} ==> {rule['consequents']}"
        text += f"(s: {rule['support']:.2f}, c: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})\n"
    ax.text(0, 0, text)
    plt.tight_layout()
    savefig(f'images/default/TOP_10_per_Min{per_metric}-{metric}.png')
    show()


def analyse_per_metric(rules: DataFrame, metric: str, metric_values: list) -> list:
    print(f'Analyse per {metric}...')
    conf = {'avg': [], 'top25%': [], 'top10': []}
    lift = {'avg': [], 'top25%': [], 'top10': []}
    top_conf = []
    top_lift = []
    nr_rules = []
    for m in metric_values:
        rs = rules[rules[metric] >= m]
        nr_rules.append(len(rs))
        conf['avg'].append(rs['confidence'].mean(axis=0))
        lift['avg'].append(rs['lift'].mean(axis=0))

        top_conf = rs.nlargest(int(0.25 * len(rs)), 'confidence')
        conf['top25%'].append(top_conf['confidence'].mean(axis=0))

        top_lift = rs.nlargest(int(0.25 * len(rs)), 'lift')
        lift['top25%'].append(top_lift['lift'].mean(axis=0))

        top_conf = rs.nlargest(10, 'confidence')
        conf['top10'].append(top_conf['confidence'].mean(axis=0))

        top_lift = rs.nlargest(10, 'lift')
        lift['top10'].append(top_lift['lift'].mean(axis=0))

    _, axs = subplots(1, 2, figsize=(10, 5), squeeze=False)
    multiple_line_chart(metric_values, conf, ax=axs[0, 0], title=f'Avg Confidence x {metric}',
                        xlabel=metric, ylabel='Avg confidence')
    print("mtric_Values: ", metric_values)
    print("mtric_Values: ", lift)

    multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                        xlabel=metric, ylabel='Avg lift')
    plt.tight_layout()

    savefig(f"images/default/{metric}_1.png")
    show()

    plot_top_rules(top_conf, 'confidence', metric)
    # savefig(f"images/default/{metric}_2.png")

    plot_top_rules(top_lift, 'lift', metric)
    # savefig(f"images/default/{metric}_3.png")

    return nr_rules


nr_rules_sp = analyse_per_metric(rules, 'support', var_min_sup)
plot_line(var_min_sup, nr_rules_sp, title='Nr rules x Support', xlabel='support', ylabel='Nr. rules', percentage=False)
plt.tight_layout()

savefig(f"images/default/Nr_rules_Support.png")

var_min_conf = [i * MIN_CONF for i in range(10, 5, -1)]
nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf)
plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules',
          percentage=False)
plt.tight_layout()

savefig(f"images/default/Nr_rules_confidence.png")
