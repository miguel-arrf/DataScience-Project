import pandas as pd
from numpy import nan
from pandas import DataFrame, read_csv, concat
from matplotlib.pyplot import figure, show, subplots, savefig
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ds_charts import plot_line, multiple_line_chart, get_variable_types
from mlxtend.frequent_patterns import apriori, association_rules

data = read_csv('../../data/NYC_collisions_tabular.csv')
data.pop("PERSON_INJURY")
data.pop("UNIQUE_ID")
data.pop("COLLISION_ID")
data.pop("VEHICLE_ID")
numeric_vars = get_variable_types(data)['Numeric']

for var in numeric_vars:
    data.pop(var)

# replace mv
tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
# replace mv

print("df columns: ", df.columns)
print("df columns: ", len(df.columns))
# transform to mining
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)

data = pd.DataFrame(te_ary, columns=te.columns_)
print("hehe: ", te.columns_)
print("hehe len: ", len(te.columns_))
# end transform to mining
print("columns: ", len(df.columns))



MIN_SUP: float = 0.00001
var_min_sup = [0.2, 0.1] + [i * MIN_SUP for i in range(100, 0, -10)]
var_min_sup = [0.3, 0.2, 0.15,0.1,  0.01]
print("var_min_sup: ", var_min_sup)
print()

print("Data: ", data)
print()

patterns: DataFrame = apriori(data, min_support=MIN_SUP, use_colnames=True, verbose=True)
print(len(patterns), 'patterns')
nr_patterns = []
for sup in var_min_sup:
    pat = patterns[patterns['support'] >= sup]
    nr_patterns.append(len(pat))

figure(figsize=(6, 4))
plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')
print("varminsup: ", var_min_sup)
print("nr patterns: ", nr_patterns)
savefig("images/default/nr_patterns_support.png")
show()

MIN_CONF: float = 0.01
print("patterns: ", patterns)
rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
print(f'\tfound {len(rules)} rules')


def plot_top_rules(rules: DataFrame, metric: str, per_metric: str) -> None:
    _, ax = subplots(figsize=(10, 3))
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
    multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                        xlabel=metric, ylabel='Avg lift')
    savefig(f"images/default/{metric}_1.png")
    show()

    plot_top_rules(top_conf, 'confidence', metric)
    # savefig(f"images/default/{metric}_2.png")

    plot_top_rules(top_lift, 'lift', metric)
    # savefig(f"images/default/{metric}_3.png")

    return nr_rules


nr_rules_sp = analyse_per_metric(rules, 'support', var_min_sup)
plot_line(var_min_sup, nr_rules_sp, title='Nr rules x Support', xlabel='support', ylabel='Nr. rules', percentage=False)
savefig(f"images/default/Nr_rules_Support.png")

var_min_conf = [i * MIN_CONF for i in range(10, 5, -1)]
nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf)
plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules',
          percentage=False)
savefig(f"images/default/Nr_rules_confidence.png")
