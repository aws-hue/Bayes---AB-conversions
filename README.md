# Bayes AB Reporter

A lightweight Python library for A/B testing analysis using Bayesian methods combined with Fisher's Exact Test. This tool helps analyze and compare the conversion rates of two variants (control vs. test) in experiments, offering insights such as Bayesian confidence intervals, hypothesis testing, and conversion rate lift visualization.

## Features

- **Bayesian Analysis**: Uses Beta distribution to estimate the posterior distribution of conversion rates for both control and test groups.
- **Fisher's Exact Test**: Provides a statistical significance test based on the contingency table for two variants.
- **Lift Calculation**: Computes the relative lift between test and control groups, with 95% credible intervals.
- **Visualizations**: Plots convergence of lift over varying sample sizes, showing the range of potential outcomes as more data is collected.
- **Flexible Grouping**: Allows grouping by different slices (e.g., country, device) to analyze different segments of the data.
  
## Installation

To install the library, simply clone the repository and install the required dependencies:

bash
git clone https://github.com/yourusername/BayesABReporter.git
cd BayesABReporter
pip install -r requirements.txt

## Usage

import pandas as pd
from BayesABReporter import BayesABReporter

# Sample DataFrame with A/B testing results

data = {
    'split': ['control', 'test'],
    'cr': [0.1, 0.12],  # Conversion rates
    'n': [1000, 1000],  # Sample sizes
    'successes': [100, 120]  # Number of conversions (optional if 'cr' is provided)
}
df = pd.DataFrame(data)

# Initialize BayesABReporter

reporter = BayesABReporter(
    df=df,
    metric='cr',
    size_col='n',
    successes_col='successes',
    variant_col='split',
    control_label='control',
    group_cols = ['region'],
    test_label='test',
    alternative='greater',  # or 'less', 'two-sided'
    target_prob=0.95,
    alpha=0.05  
)

# Run analysis
result = reporter.analyze()

print(result)

# Plot convergence
reporter.plot_convergence('control', 'test')
Parameters
df: DataFrame containing the experiment data. Must contain columns for conversion rates, sample sizes, and successes.

metric: Column with the conversion rate (e.g., 'cr'). Ignored if successes_col is provided.

size_col: Column for the sample size (e.g., 'n').

successes_col: Column containing the number of conversions (optional if metric is used).

variant_col: Column indicating the experimental variant (e.g., 'split').

control_label: Label for the control group (default 'control').

test_label: Label for the test group (default 'test').

alternative: Type of hypothesis test ('greater', 'less', 'two-sided').

target_prob: Confidence threshold for Bayesian analysis (default 0.95).

alpha: Significance level for Fisher's exact test (default 0.05).

# Methods
analyze(): Runs the A/B test analysis and returns a tidy DataFrame with the results, including Bayesian conversion rate estimates, Fisher's p-value, lift, and significance.

plot_convergence(): Plots the convergence of lift between the control and test groups as sample size increases.

## Requirements
Python 3.x

numpy

pandas

scipy
matplotlib


## Acknowledgments
The approach is based on Bayesian statistics and Fisher's Exact Test for hypothesis testing.
