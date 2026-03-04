from NASProblem import Problem, NASProblem
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


## 3 available datasets
## cifar10, cifar100, ImageNet16-120
# prob1 = NASProblem(dataset="cifar100")
# print(prob1)
# print(prob1.data.shape)

# prob1.data  # prblem dataset
# prob1.data[prob1.var_cols]  # decision space
# prob1.data[prob1.obj_cols]  # objective space

# print(prob1.evaluate(np.array([["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none", "none"],
#                                ["none", "none", "skip_connect", "nor_conv_1x1", "n", "none"]
#                                ])))



## 5 available options
## "avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"
option_config = [
    ["skip_connect", "nor_conv_3x3"],
    ["skip_connect", "none"],
    ["avg_pool_3x3", "nor_conv_3x3", "nor_conv_1x1", "none"],
    ["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"],
    ["avg_pool_3x3", "skip_connect", "nor_conv_1x1"],
    ["skip_connect", "none"]
]
prob2 = NASProblem(dataset='ImageNet16-120', option_config=option_config)
print(prob2)
print(prob2.data.shape)

## apply non-dominated sorting on prob2
pf_indices = NonDominatedSorting().do(
        prob2.data[prob2.obj_cols].to_numpy(), only_non_dominated_front=True
    )

## pareto solutions
print(len(pf_indices))
# print(prob2.data.iloc[pf_indices])
# print(prob2.data[prob2.obj_cols].iloc[pf_indices])
