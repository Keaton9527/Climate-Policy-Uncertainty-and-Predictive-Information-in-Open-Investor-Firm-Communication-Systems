# Release validation

Validation performed on 29 October 2026 against the local study inputs and saved results. Private inputs and calculated outputs remain outside the public repository. Python 3.12 and the pinned packages in `requirements.txt` were used.

| Check | Result |
| --- | --- |
| Baseline sample and target construction | 61 stocks and 108,244 modeling firm-days |
| Table 2 retained descriptions | All three measures in all three regimes agree with the original experiment 008 results |
| Ranking evaluation | Recomputed IC/RankIC from all 120 dynamic prediction files; return signs and forecast keys checked |
| Table 4 | All ten model/feature rows agree on MCC, AUC, IC, and RankIC with experiment 012 |
| Table 3 | All 15 technical-candidate selection summaries agree with experiment 008 |
| Table 6 regime/source results | All 12 regime/feature rows agree on MCC, AUC, IC, and RankIC with experiment 013 |
| Tables A4 and A5 | All six cross-model regime rows and ten fixed-feature rows agree with experiment 008 |
| Table A7 | All nine estimates, confidence bounds, p-values, and Holm-adjusted p-values agree with experiment 021 after 5,000 bootstrap repetitions |
| Table 7 and cross-metric sensitivity | All 36 contrasts agree with experiment 015 on estimates, confidence bounds, p-values, effective repetitions, and the 9/36-family corrections |
| Residual CCPU | Reply-side MCC contrast agrees with experiment 016 on estimate, confidence bounds, and p-value |
| Probability-loss regression | All 12 reported term/lag rows agree with experiment 016 on coefficients, standard errors, and p-values |
| Firm exposure | Transition measure reconstruction and disclosure alignment pass; both within-exposure contrasts and the signed difference in differences agree with experiment 017 run 002 |
| Representative training reproduction | Recomputed all five selection methods for window 7; selected J, ROCP, K, and RSI match the saved selection. Retrained reply-side XGBoost; all 7,619 test probabilities and classifications exactly match the saved forecasts |
| Public numerical tests | Seven tests cover weighted AUC with ties, block resampling, Holm correction, report availability, forecast pairing, per-stock indicators, and daily text alignment/late-reply recoding |
| Static and release checks | All added Python files parse; YAML/JSON files parse; public input paths are configurable; added files contain no private datasets or credentials |

CSV comparisons use relative tolerance 1e-9 and absolute tolerance 1e-11 to accommodate printed precision. The representative prediction check uses exact array equality. The statistical comparisons use seed 42, 5,000 repetitions, and 20-date blocks, matching the manuscript.

The release validation recomputed downstream statistics from the original frozen forecasts and retrained the window-7 reply-side XGBoost model. It did not retrain every model/window combination or the language models, or regenerate the complete historical text/sentiment panel. The optional preparation routines have numerical/CLI checks; exact historical forecasts use the supplied study baseline.
