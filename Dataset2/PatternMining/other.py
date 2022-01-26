MIN_SUP: float = 0.8
var_min_sup = [0.3, 0.4] + [i * MIN_SUP for i in range(10, 0, -5)] # antes era -10 e nao 5
var_min_sup = sorted(var_min_sup)

var_min_sup = [0.1, 0.2, 0.6, 0.8]

print(var_min_sup)