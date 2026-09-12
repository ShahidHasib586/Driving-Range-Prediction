# Electric vehicle driving range prediction

A collection of regression experiments and research working files for electric vehicle driving range prediction. The repository contains model scripts, datasets, plots, and manuscript/conference preparation assets.

## Experiment map

| Folder | Contents |
| --- | --- |
| `Multiple Linear Regression with K-fold Cross validation/` | Multiple linear regression with cross validation. |
| `Random forest Regression/` | Random forest regression scripts and outputs. |
| `Support Vector Regression/` | Support vector regression experiments. |
| `Simple_linear Regression/` | Linear regression analysis. |
| `poly_nomial Regression/` | Polynomial regression experiments. |
| `Principle Component Analysis/` | Feature analysis and PCA working files. |

## Getting started

Choose an experiment and inspect its imports, dataset paths, and target variable before running the script in that folder. These are research scripts with experiment specific settings rather than a packaged application. Files with spaces in their names must be quoted when launched from a terminal.

Use an isolated Python environment and install the dependencies imported by the selected experiment. Some files depend on older library APIs and local dataset paths, so environment adaptation may be required.

## Interpreting results

Compare models using the same preprocessing, data split, and evaluation protocol. Report error units and the target definition alongside RMSE or other metrics, and avoid fitting preprocessing on held out evaluation data. Saved figures are historical experiment outputs and do not establish a universal range prediction accuracy.

The repository's existing folder names are retained so links to earlier work continue to resolve.
