import pandas as pd
import scipy.stats as stats
import numpy as np
from tabulate import tabulate

# File input
file = pd.read_csv("C:\\Users\\rlu\\Documents\\Collected Data\\genotype.csv")

# Basic Parameter Input (Need editing)
df = file 
xfac = "Samples"
res = "C. Angles"
evar = False
alpha = 0.05
test_type = 2

def Welch_T_test():
        # drop NaN
        df = file.dropna()
        if df.shape[0] < 2:
            raise Exception("Very few observations to run t-test")
        if alpha < 0 or alpha > 1:
            raise Exception("alpha value must be in between 0 and 1")
        
        if test_type == 2:
            if xfac and res is None:
                raise Exception("xfac or res variable is missing")
            if res not in df.columns or xfac not in df.columns:
                raise ValueError("res or xfac column is not in dataframe")
            
            levels = df[xfac].unique()
            levels.sort()
            
            a_val = df.loc[df[xfac] == levels[0], res].to_numpy()
            b_val = df.loc[df[xfac] == levels[1], res].to_numpy()
            a_count, b_count = len(a_val), len(b_val)
            count = [a_count, b_count]
            mean = [df.loc[df[xfac] == levels[0], res].mean(), df.loc[df[xfac] == levels[1], res].mean()]
            sem = [df.loc[df[xfac] == levels[0], res].sem(), df.loc[df[xfac] == levels[1], res].sem()]
            sd = [df.loc[df[xfac] == levels[0], res].std(), df.loc[df[xfac] == levels[1], res].std()]
            ci = (1-alpha)*100
            # degree of freedom
            # a_count, b_count = np.split(count, 2)
            dfa = a_count - 1
            dfb = b_count - 1
            # sample variance
            with np.errstate(invalid='ignore'):
                var_a = np.nan_to_num(np.var(a_val, ddof=1))
                var_b = np.nan_to_num(np.var(b_val, ddof=1))
            mean_diff = mean[0] - mean[1]
            # variable 95% CI
            varci_low = []
            varci_up = []
            tcritvar = [(stats.t.ppf((1 + (1-alpha)) / 2, dfa)), (stats.t.ppf((1 + (1-alpha)) / 2, dfb))]
            for i in range(len(levels)):
                varci_low.append(mean[i] - (tcritvar[i] * sem[i]))
                varci_up.append(mean[i] + (tcritvar[i] * sem[i]))

            if evar is False:
                # Welch's t-test for unequal variance
                message = 'Two sample t-test with unequal variance (Welch\'s t-test)'
                if a_count == 1 or b_count == 1:
                    raise Exception('Not enough observation for either levels. The observations should be > 1 for both levels')
                a_temp = var_a / a_count
                b_temp = var_b / b_count
                dfr = ((a_temp + b_temp) ** 2) / ((a_temp ** 2) / (a_count - 1) + (b_temp ** 2) / (b_count - 1))
                rounded_dfr = int(dfr)
                se = np.sqrt(a_temp + b_temp)

            tval = np.divide(mean_diff, se)
            oneside_pval = stats.t.sf(np.abs(tval), dfr)
            twoside_pval = oneside_pval * 2
            # 95% CI for diff
            tcritdiff = stats.t.ppf((1 + (1-alpha)) / 2, dfr)
            diffci_low = mean_diff - (tcritdiff * se)
            diffci_up = mean_diff + (tcritdiff * se)

            # print results
            print(message)
            print(tabulate([["Mean diff", mean_diff], ["t", tval], ["Std Error", se], ["df", dfr], ["rounded df", rounded_dfr],
                                     ["P-value (one-tail)", oneside_pval], ["P-value (two-tail)", twoside_pval],
                                     ["Lower "+str(ci)+"%", diffci_low], ["Upper "+str(ci)+"%", diffci_up]]) + '\n')
                
            print('Parameter estimates\n\n' + tabulate([[levels[0], count[0], mean[0], sd[0], sem[0], varci_low[0],
                                      varci_up[0]], [levels[1], count[1], mean[1], sd[1], sem[1],
                                                     varci_low[1], varci_up[1]]],
                                    headers=["Samples", "Number", "Mean", "Std Dev", "Std Error",
                                             "Lower "+str(ci)+"%", "Upper "+str(ci)+"%"]) + '\n')

            df5_1_to_100 = [6.314, 2.920, 2.353, 2.132, 2.015,
                        1.943, 1.895, 1.860, 1.833, 1.812,
                        1.796, 1.782, 1.771, 1.761, 1.753,
                        1.746, 1.740, 1.734, 1.729, 1.725,
                        1.721, 1.717, 1.714, 1.711, 1.708,
                        1.706, 1.703, 1.701, 1.699, 1.697,
                        1.696, 1.694, 1.692, 1.691, 1.690,
                        1.688, 1.687, 1.686, 1.685, 1.684,
                        1.683, 1.682, 1.681, 1.680, 1.679,
                        1.679, 1.678, 1.677, 1.677, 1.676,
                        1.675, 1.675, 1.674, 1.674, 1.673,
                        1.673, 1.672, 1.672, 1.671, 1.671,
                        1.670, 1.670, 1.669, 1.669, 1.669,
                        1.668, 1.668, 1.668, 1.667, 1.667,
                        1.667, 1.666, 1.666, 1.666, 1.665,
                        1.665, 1.665, 1.665, 1.664, 1.664,
                        1.664, 1.664, 1.663, 1.663, 1.663,
                        1.663, 1.663, 1.662, 1.662, 1.662,
                        1.662, 1.662, 1.661, 1.661, 1.661,
                        1.661, 1.661, 1.661, 1.660, 1.660]

            crit_val = (df5_1_to_100[int(dfr) - 1]) * (-1)
            print("Critical Value: ", crit_val)

            '\n'

            if tval < crit_val:
                print("Result: ", tval, "<", crit_val)
                print("")
                print("Null Hypothesis: Accepted")
                print("Regular cardboard is better than Pa-Bi-Board.")

            else:
                print("Result: ", tval, ">=", crit_val)
                print("")
                print("Null Hypothesis: Rejected")
                print("Alternatibe Hypothesis: Accepted")
                print("Pa-Bi-Board is equal or better than regular cardboard.")

Welch_T_test()
