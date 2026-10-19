# Tabla de vidas medias (DESCRIPTIVE / FIXTURE / CLOCK_UNVERIFIED / FIXTURE_RESEARCH_ONLY)

| familia | n | ratio vol. barra de publicación | reloj | vida media vol. (h) [IC 95%] | vida media tendencia (h) [IC 95%] | ¿difieren? | veredicto deriva | pendiente 4h por z | p |
|---|---|---|---|---|---|---|---|---|---|
| United States — Nonfarm Payrolls | 90 | 3.31 | CLOCK_CONSISTENT | 0.81 [0.47, 1.20] | 0.33 [0.25, 0.58] | DIFFER | NO_DRIFT_DETECTED | -9.70e-04 | 0.06 |
| United States — CPI | 145 | 1.84 | CLOCK_CONSISTENT | 0.57 [0.25, 1.89] | 0.25 [0.25, 0.40] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -2.65e-04 | 0.32 |
| United States — Core CPI | 145 | 1.84 | CLOCK_CONSISTENT | 0.57 [0.25, 1.89] | 0.25 [0.25, 0.47] | NOT_DISTINGUISHABLE | SURPRISE_CONDITIONAL_DRIFT | -6.40e-04 | 0.01 |
| United States — Retail Sales | 92 | 2.23 | CLOCK_MISMATCH | 0.35 [0.25, 0.95] | 0.25 [0.25, 0.93] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -5.90e-04 | 0.07 |
| United States — Core Retail Sales | 80 | 2.22 | CLOCK_MISMATCH | 0.28 [0.25, 1.01] | 0.25 [0.25, 0.60] | NOT_DISTINGUISHABLE | SURPRISE_CONDITIONAL_DRIFT | -8.94e-04 | 0.00 |
| United States — ISM Manufacturing PMI | 87 | 1.62 | CLOCK_CONSISTENT | 0.41 [0.25, 1.30] | 0.25 [0.25, 0.87] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -5.45e-04 | 0.04 |
| United States — GDP | 64 | 1.88 | CLOCK_MISMATCH | 1.44 [0.50, 2.27] | 0.25 [0.25, 1.59] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -3.49e-04 | 0.20 |
| United States — Unemployment Rate | 90 | 3.31 | CLOCK_CONSISTENT | 0.81 [0.47, 1.20] | 0.38 [0.25, 1.73] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -4.74e-04 | 0.36 |
| United States — ADP Nonfarm Employment Change | 68 | 2.00 | CLOCK_CONSISTENT | 0.28 [0.25, 0.57] | 0.25 [0.25, 1.81] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | -3.86e-04 | 0.21 |
| United States — Fed Interest Rate Decision | 23 | 4.38 | CLOCK_CONSISTENT | 0.60 [0.27, 1.13] | n/d n/d | NOT_COMPARABLE | NOT_FITTED | n/d | n/d |
| Euro Zone — CPI | 229 | 0.86 | CLOCK_CONSISTENT | 19.09 [10.17, 24.00] | 0.25 [0.25, 12.00] | DIFFER | NO_DRIFT_DETECTED | -8.78e-05 | 0.62 |
| Germany — German ZEW Economic Sentiment | 111 | 1.10 | CLOCK_MISMATCH | 3.32 [0.25, 8.01] | 0.25 [0.25, 12.00] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | 1.83e-04 | 0.48 |
| Germany — German Ifo Business Climate Index | 83 | 1.52 | CLOCK_CONSISTENT | 0.25 [0.25, 24.00] (censurada) | 12.00 [0.25, 12.00] | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | 8.67e-05 | 0.74 |
| Germany — German Manufacturing PMI | 154 | 1.79 | CLOCK_CONSISTENT | 0.25 [0.25, 0.37] (censurada) | 0.45 [0.25, 0.86] | NOT_DISTINGUISHABLE | UNCONDITIONAL_DRIFT_PRESENT | 8.77e-04 | 0.00 |

## Selección de eventos (periodo de decisión, FDR Benjamini-Hochberg)

| familia | estado | n entren. | n decisión | dirección | media OOS (log) | efecto/riesgo OOS | p OOS | BH | seleccionada |
|---|---|---|---|---|---|---|---|---|---|
| United States — Nonfarm Payrolls | OK | 75 | 28 | -1 | 1.08e-03 | 0.479 | 0.011 | False | False |
| United States — ADP Nonfarm Employment Change | OK | 76 | 28 | -1 | 6.80e-04 | 0.361 | 0.056 | False | False |
| Germany — German Manufacturing PMI | OK | 155 | 56 | +1 | 7.96e-04 | 0.303 | 0.023 | False | False |
| United States — Core Retail Sales | OK | 75 | 28 | -1 | 5.40e-04 | 0.174 | 0.357 | False | False |
| United States — CPI | OK | 142 | 54 | -1 | 3.66e-04 | 0.138 | 0.312 | False | False |
| United States — ISM Manufacturing PMI | OK | 76 | 28 | -1 | 3.49e-04 | 0.119 | 0.529 | False | False |
| United States — Core CPI | OK | 142 | 54 | -1 | 3.19e-04 | 0.111 | 0.414 | False | False |
| United States — Retail Sales | OK | 75 | 28 | -1 | 2.68e-04 | 0.083 | 0.660 | False | False |
| United States — Unemployment Rate | OK | 75 | 28 | -1 | 1.12e-04 | 0.063 | 0.741 | False | False |
| Germany — German Ifo Business Climate Index | OK | 76 | 29 | +1 | 1.02e-04 | 0.040 | 0.830 | False | False |
| Euro Zone — CPI | OK | 212 | 85 | -1 | -3.93e-05 | -0.032 | 0.770 | False | False |
| United States — GDP | OK | 76 | 27 | -1 | -1.64e-04 | -0.075 | 0.696 | False | False |
| Germany — German ZEW Economic Sentiment | OK | 75 | 28 | -1 | -1.11e-03 | -0.353 | 0.061 | False | False |
| United States — Fed Interest Rate Decision | TOO_FEW_EVENTS | 0 | 0 | | | | | | |
