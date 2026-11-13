# Rule hit rates (1994 training year; validation 1995 for stability)

Base fraud rate in training: 0.067. Cost break-even precision 1/(1+10) = 0.091.

| rule | fires | fire_rate | fraud_share | lift | fraud_caught | valid_fraud_share | valid_lift | break_even_ok | decision |
|---|---|---|---|---|---|---|---|---|---|
| R01_EARLY_ACCIDENT | 42 | 0.007 | 0.119 | 1.787 | 5 | 0.089 | 1.534 | True | keep |
| R02_NO_EVIDENCE | 5976 | 0.973 | 0.067 | 1.013 | 403 | 0.058 | 1.004 | False | drop |
| R03_RECENT_ADDRESS_CHANGE | 188 | 0.031 | 0.181 | 2.715 | 34 | 0.123 | 2.129 | True | keep |
| R04_FREQUENT_CLAIMANT | 765 | 0.125 | 0.024 | 0.353 | 18 | 0.043 | 0.734 | False | drop |
| R05_MANY_SUPPLEMENTS | 1481 | 0.241 | 0.054 | 0.811 | 80 | 0.054 | 0.936 | False | drop |
| R06_COVERAGE_MISMATCH | 1307 | 0.213 | 0.116 | 1.735 | 151 | 0.093 | 1.602 | True | keep |
| R07_EARLY_CLAIM | 31 | 0.005 | 0.129 | 1.937 | 4 | 0.143 | 2.466 | True | keep |
