# Dataset Card: UCI Covertype

- Dataset: Covertype
- Creator: Jock Blackard
- Source: UCI Machine Learning Repository, dataset 31
- URL: https://archive.ics.uci.edu/dataset/31/covertype
- DOI: https://doi.org/10.24432/C50K5N
- License: CC BY 4.0
- Shape: 581,012 observations, 54 inputs, 7 forest-cover classes
- Missing values reported by UCI: none
- Quick-run sample: 12,000 rows, selected with stratification and seed 42

The first 10 quantitative columns are scaled using statistics fitted on the training split only. The 44 binary wilderness/soil indicators are left unscaled. The target is removed before preprocessing. Random stratification preserves class proportions but does not test spatial transfer between geographic areas.

Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
