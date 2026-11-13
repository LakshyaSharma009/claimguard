# Fairness check: with vs without Sex and Age

Same model (hgb, same parameters). Protected columns: Sex, Age, AgeOfPolicyHolder.

| variant | valid_pr_auc | test_pr_auc |
|---|---|---|
| with Sex/Age | 0.218 | 0.151 |
| without Sex/Age | 0.192 | 0.136 |

PR-AUC gap on validation (with minus without) = +0.0255. Shipped model: **without Sex/Age**, set by `use_protected_attributes` in config/thresholds.yaml.


How the decision was made: the first version of this check used a PR-AUC-only rule (keep the columns if they add more than 0.01) and kept them. The per-group table below then showed that the model with Sex/Age flags women at about half the rate of men, and catches far less of their fraud. Many regulators also forbid sex-based insurance decisions. The columns are now excluded as a policy, and this report shows what that costs.


Flag rate per group when the top 10% of 1996 claims are reviewed:


## with Sex/Age

| attribute | group | claims | fraud_rate | flag_rate | recall |
|---|---|---|---|---|---|
| Sex | Female | 629 | 0.041 | 0.056 | 0.115 |
| Sex | Male | 3454 | 0.054 | 0.108 | 0.267 |
| Age | 26-40 | 2017 | 0.054 | 0.132 | 0.324 |
| Age | 41-65 | 1678 | 0.051 | 0.057 | 0.151 |
| Age | 65+ | 149 | 0.027 | 0.034 | 0.000 |
| Age | <=25 | 169 | 0.083 | 0.095 | 0.286 |
| Age | unknown | 70 | 0.014 | 0.386 | 1.000 |

## without Sex/Age

| attribute | group | claims | fraud_rate | flag_rate | recall |
|---|---|---|---|---|---|
| Sex | Female | 629 | 0.041 | 0.095 | 0.346 |
| Sex | Male | 3454 | 0.054 | 0.105 | 0.257 |
| Age | 26-40 | 2017 | 0.054 | 0.111 | 0.296 |
| Age | 41-65 | 1678 | 0.051 | 0.083 | 0.221 |
| Age | 65+ | 149 | 0.027 | 0.047 | 0.000 |
| Age | <=25 | 169 | 0.083 | 0.107 | 0.357 |
| Age | unknown | 70 | 0.014 | 0.471 | 1.000 |
