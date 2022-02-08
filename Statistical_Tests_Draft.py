# Libraries
import pandas as pd
import scipy.stats as stats
import numpy as np
from tabulate import tabulate

print("")

# File input (Need editing) 
# [DOUBLE SLAHES '\\' FOR LOCAL FILES (not URLs) TO AVOUD UNICODE ESCAPE ERRORS]
csv_file = "https://raw.githubusercontent.com/MaybeLance/Data-Analysis-for-Cardboard-Development/main/Sample%20Data/Germination%20Sample%20Data.csv"
data = pd.read_csv(csv_file)

# Basic Parameter Input (Need editing)
df = data 
xfac = "Samples"
res = "Seeds"
alpha = 0.05

# Warnings due to wrong inputs
df = data.dropna()
if df.shape[0] < 2:
    raise Exception("Very few observations to run t-test")
if alpha < 0 or alpha > 1:
    raise Exception("alpha value must be in between 0 and 1")
if xfac and res is None:
    raise Exception("xfac or res variable is missing")
if res not in df.columns or xfac not in df.columns:
    raise ValueError("res or xfac column is not in dataframe")

# Get data drom CSV file         
levels = df[xfac].unique()
levels.sort()
a_val = df.loc[df[xfac] == levels[0], res].to_numpy()
b_val = df.loc[df[xfac] == levels[1], res].to_numpy()

# Parameter Estimates
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

# Sample Variance
with np.errstate(invalid='ignore'):
    var_a = np.nan_to_num(np.var(a_val, ddof=1))
    var_b = np.nan_to_num(np.var(b_val, ddof=1))

mean_diff = mean[0] - mean[1]

# variable 95% Confidence Interval
varci_low = []
varci_up = []
tcritvar = [(stats.t.ppf((1 + (1-alpha)) / 2, dfa)), (stats.t.ppf((1 + (1-alpha)) / 2, dfb))]
for i in range(len(levels)):
    varci_low.append(mean[i] - (tcritvar[i] * sem[i]))
    varci_up.append(mean[i] + (tcritvar[i] * sem[i]))

# Print Parameter Results
print('Descriptive Statistics\n' + tabulate([[levels[0], count[0], mean[0], sd[0], sem[0], varci_low[0], varci_up[0]],
                                            [levels[1], count[1], mean[1], sd[1], sem[1], varci_low[1], varci_up[1]]],
                                    headers=["Samples", "Number", "Mean", "Std Dev", "Std Error",
                                            "Lower "+str(ci)+"%", "Upper "+str(ci)+"%"]) + '\n')



# Test for Normality by Shapiro-Wilk Test
# For Sample A
a_stat, a_p = stats.shapiro(a_val)

# interpret A
a_alpha = 0.05
if a_p > a_alpha:
	behave_a = 'Normal Distribution (fail to reject H0)'
	sw_a_norm = True
else:
	behave_a = 'Non-Normal Distribution (reject H0)'
	sw_a_norm = False

# For Sample B
b_stat, b_p = stats.shapiro(b_val)

# interpret B
b_alpha = 0.05
if b_p > b_alpha:
	behave_b = 'Normal Distribution (fail to reject H0)'
	sw_b_norm = True
else:
	behave_b = 'Non-Normal Distribution (reject H0)'
	sw_b_norm = False

# Verdict
# If both normal
if sw_a_norm == True and sw_b_norm == True:
	sw_verdict = "Parametric Route"
	sw_bool_result = True
	# Version of Levene
	leve_version = 'mean'

# If at least one non-normal
else:
	sw_verdict = "Non-Parametric Route"
	sw_bool_result = False
	# Version of Levene
	leve_version = 'median'

# Print results SHAPIRO
print('Shapiro-Wilk Normality Test Results\n' + tabulate([[levels[0], a_stat, a_p, behave_a, sw_verdict],
                                            				[levels[1], b_stat, b_p, behave_b]],
                                    				headers=["Groups", "Test Statistics", "P-Value", "Behavior", "Verdict"]) + '\n')

# Test for Equal Variance by Levene's Test
leve_stat, leve_p = stats.levene(a_val, b_val, center = leve_version)

# Interpret and Verdict Levene
leve_alpha = 0.05
if leve_p > leve_alpha:
	leve_result = 'Have equal variances (fail to reject H0)'
	leve_verdict = "Equal Route"
	leve_bool_result = True
else:
	leve_result = 'Have unequal variances (reject H0)'
	leve_verdict = "Unequal Route"
	leve_bool_result = False

# Number of Arrays
num_of_groups = len(stats.levene(a_val, b_val))
if num_of_groups == 2:
	num_of_groups = 'Two'

# Print results LEVENE
print('Levene Equal Variance Test Results\n' + tabulate([[num_of_groups, leve_stat, leve_p, leve_result, leve_verdict]],
                                    			headers=["Groups", "Test Statistics", "P-Value", "Behavior", "Verdict"]) + '\n')



# BIG 4 STATISTICAL TESTS

# Parametric Route AND Homoscedasticity
def Student_T_test():
    message = "Student's T-Test"
    p_var = (dfa * var_a + dfb * var_b) / (dfa + dfb)
    # std error
    se = np.sqrt(p_var * (1.0 / a_count + 1.0 / b_count))
    dfr = dfa + dfb

    tval = np.divide(mean_diff, se)
    trash_1, oneside_pval = stats.ttest_ind(a_val, b_val, alternative = 'less')
    trash_2, twoside_pval = stats.ttest_ind(a_val, b_val, alternative = 'two-sided')
    # 95% CI for diff
    tcritdiff = stats.t.ppf((1 + (1-alpha)) / 2, dfr)
    diffci_low = mean_diff - (tcritdiff * se)
    diffci_up = mean_diff + (tcritdiff * se)

    # print results
    print(message)
    print(tabulate([["Mean diff", mean_diff], 
                    ["t", tval], 
                    ["Std Error", se], 
                    ["df", dfr],
                    ["P-value (one-tail)", oneside_pval], 
                    ["P-value (two-tail)", twoside_pval], 
                    ["Lower "+str(ci)+"%", diffci_low], 
                    ["Upper "+str(ci)+"%", diffci_up]]) + '\n')



# Parametric Route AND Heteroscedasticity
def Welch_T_test():
    message = "Welch-Satterthwaite T-Test"
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
    print(tabulate([["Mean diff", mean_diff], 
                    ["t", tval], 
                    ["Std Error", se], 
                    ["df", dfr], 
                    ["rounded df", rounded_dfr],
                    ["P-value (one-tail)", oneside_pval], 
                    ["P-value (two-tail)", twoside_pval], 
                    ["Lower "+str(ci)+"%", diffci_low], 
                    ["Upper "+str(ci)+"%", diffci_up]]) + '\n')



# Non-Parametric Route AND Homoscedasticity
def Mann_Whitney_Test():
    message = "Wilcoxon-Mann-Whitney U Test"
    mw_stats, mw_one_p_val = stats.mannwhitneyu(a_val, b_val, 
                                                alternative = 'less')
    
    # Get two-tailed p-value
    trash, mw_two_p_val = stats.mannwhitneyu(a_val, b_val,
                                                alternative = 'two-sided')

    # print results
    print(message)
    print(tabulate([["Mean diff", mean_diff], 
                    ["Statistics", mw_stats],
                    ["P-value (one-tail)", mw_one_p_val], 
                    ["P-value (two-tail)", mw_two_p_val]]) + '\n')



# Non-Parametric Route AND Heteroscedasticity
def Brunner_Munzel_Test():
    message = "Brunner-Munzel Test"
    bm_stats, bm_one_p_val = stats.brunnermunzel(a_val, b_val, 
                                                alternative = 'less')

    trash, bm_two_p_val = stats.brunnermunzel(a_val, b_val, 
                                                alternative = 'two-sided')
    
    # print results
    print(message)
    print(tabulate([["Mean diff", mean_diff], 
                    ["Statistics", bm_stats],
                    ["P-value (one-tail)", bm_one_p_val], 
                    ["P-value (two-tail)", bm_two_p_val]]) + '\n')



# Determining Stats Code Block
def statistics_choosing():
    if sw_bool_result == True and leve_bool_result == True:
        return Student_T_test()

    elif sw_bool_result == True and leve_bool_result == False:
        return Welch_T_test()

    elif sw_bool_result == False and leve_bool_result == True:
        return Mann_Whitney_Test()

    elif sw_bool_result == False and leve_bool_result == False:
        return Brunner_Munzel_Test()

statistics_choosing()



print("------------------ End of Data Analysis ------------------")
