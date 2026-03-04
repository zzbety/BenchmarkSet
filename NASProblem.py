import numpy as np
import pandas as pd


class Problem:

    def __init__(self,
                 n_var,
                 n_obj,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 n_var_cont=0,
                 n_var_cat=0,
                 xl=None,
                 xu=None,
                 xlels=None):

        # Basic info
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr
        self.n_constr = n_ieq_constr + n_eq_constr

        # Variable type info
        self.n_var_cont = n_var_cont
        self.n_var_cat = n_var_cat

        # Continuous bounds
        self.xl = xl
        self.xu = xu

        # Categorical levels
        self.xlels = xlels


    def check_bound(self, X):

        n_samples, n_vars = X.shape
        V = np.zeros((n_samples, self.n_var), dtype=int)    # 0 = satified

        for i in range(n_samples):
            xi = X[i]

            # ---- Continuous ----
            for j in range(self.n_var_cont):
                if self.xl is not None and xi[j] < self.xl[j]:
                    V[i, j] = 1
                if self.xu is not None and xi[j] > self.xu[j]:
                    V[i, j] = 1

            # ---- Categorical ----
            for j in range(self.n_var_cont, self.n_var_cont + self.n_var_cat):
                level_list = self.xlels[j - self.n_var_cont]
                if xi[j] not in level_list:
                    V[i, j] = 1
               
        return V


    def evaluate(self, X):
        pass

    def __str__(self):
        return f"""
        -----------------
        n_var: {self.n_var}
        (cont, cat): ({self.n_var_cont}, {self.n_var_cat})
        n_obj: {self.n_obj}
        n_constr: {self.n_constr}
        -----------------
        """
    

class NASProblem(Problem):

    def __init__(self,
                 dataset="cifar10",               # choose cifar10 / cifar100 / ImageNet16-120
                 option_config=None               # list of allowed options per variable
                 ):

        # 6 categorical variables, 4 objectives
        n_var = 6
        n_obj = 4

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,
            n_eq_constr=0,
            n_var_cont=0,
            n_var_cat=6
        )

        # -------- Load dataset --------
        self.dataset_name = dataset
        self.full_data = pd.read_csv(f"nasbench201_{dataset}_all.csv")

        # decision variable column names
        self.var_cols = [f"hp_x{i}" for i in range(6)]

        # objective column names
        self.obj_cols = [
            "metric_valid_error",
            "metric_runtime",
            "metric_latency",
            "metric_flops"
        ]

        # default full level set 
        full_levels = [
            "avg_pool_3x3",
            "nor_conv_3x3",
            "skip_connect",
            "nor_conv_1x1",
            "none"
        ]

        if option_config is None:
            self.xlels = [full_levels for i in range(6)]
        else:
            if len(option_config) != 6:
                raise ValueError("option_config must have length 6.")
            self.xlels = option_config

    @property
    def data(self):
        mask = np.ones(len(self.full_data), dtype=bool)

        for j in range(self.n_var):
            allowed = self.xlels[j]
            mask &= self.full_data[self.var_cols[j]].isin(allowed)

        return self.full_data[mask]


    def evaluate(self, X):

        V = self.check_bound(X)

        F = []
        G = []

        for i in range(X.shape[0]):

            xi = X[i]
            G.append(xi.tolist())

            if np.any(V[i] == 1):
                # invalid solution
                F.append([np.nan] * self.n_obj)
                continue

            mask = (self.data[self.var_cols] == xi).all(axis=1)

            matched = self.data[mask]

            if len(matched) == 0:
                # architecture not found in dataset
                F.append([np.nan] * self.n_obj)
            else:
                F.append(matched[self.obj_cols].values[0])

        return np.array(F), G, V


    def __str__(self):
        return f"""
        -----------------------------
        Dataset: {self.dataset_name}
        n_var: {self.n_var}
        n_obj: {self.n_obj}
        Levels per variable: {[len(levels) for levels in self.xlels]}
        -----------------------------
        """