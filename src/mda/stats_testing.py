import numpy as np
import pandas as pd
from scipy import stats
import json
import os
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

import warnings
warnings.filterwarnings('ignore')

def test_normality(sample, alpha=0.05):
    """Test residuals for normality using multiple tests
    
    Parameters:
    -----------
    residuals : array-like
        The residuals (y_true - y_pred) to test for normality
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    dict
        Dictionary with test results and overall conclusion
    """

    # Shapiro-Wilk test 
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    
    # D'Agostino's K^2 test
    dagostino_stat, dagostino_p = stats.normaltest(sample)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(sample, 'norm')
    # Use the 5% level (index 2)
    anderson_stat = anderson_result.statistic
    anderson_crit = anderson_result.critical_values[2]
    anderson_sig = anderson_stat > anderson_crit
    
    # Compile results
    results = {
        'Shapiro-Wilk Test': {'statistic': shapiro_stat, 'p/critical value': shapiro_p, 'normal': shapiro_p > alpha},
        'D\'Agostino\'s K^2 Test': {'statistic': dagostino_stat, 'p/critical value': dagostino_p, 'normal': dagostino_p > alpha},
        'Anderson-Darling Test': {'statistic': anderson_stat, 'p/critical value': anderson_crit, 'normal': not anderson_sig}
    }
    
    # Overall conclusion (at least 2 tests agree)
    normal_votes = sum([results[test]['normal'] for test in results])
    results['conclusion'] = normal_votes >= 2
    
    return results

def test_homoscedasticity(predictors, residuals, alpha=0.05):
    """Test residuals for homoscedasticity using multiple tests
    
    Parameters:
    -----------
    predictors : array-like
        The feature values
    residuals : array-like
        The residuals (y_true - y_pred)
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    dict
        Dictionary with test results and overall conclusion
    """
    # Set up the model matrix X (with predicted values)
    X = sm.add_constant(predictors)
    
    # Breusch-Pagan test
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
    
    # White's test
    white_stat, white_p, _, _ = het_white(residuals, X)
    
    # Goldfeld-Quandt test
    gq_stat, gq_p, _ = het_goldfeldquandt(residuals, X)
    
    # Compile results
    results = {
        'Breusch-Pagan Test': {'statistic': bp_stat, 'p-value': bp_p, 'homoscedastic': bp_p > alpha},
        'White\'s Test': {'statistic': white_stat, 'p-value': white_p, 'homoscedastic': np.isnan(white_p) or white_p > alpha},
        'Goldfeld-Quandt Test': {'statistic': gq_stat, 'p-value': gq_p, 'homoscedastic': np.isnan(gq_p) or gq_p > alpha}
    }
    
    # Overall conclusion (at least 2 tests agree)
    valid_tests = [test for test in results if not np.isnan(results[test]['p-value'])]
    if len(valid_tests) >= 2:
        homo_votes = sum([results[test]['homoscedastic'] for test in valid_tests])
        results['conclusion'] = homo_votes >= len(valid_tests)/2
    else:
        results['conclusion'] = None  # Not enough valid tests
    
    return results

def test_statistical_significance(input, alpha=0.05):
    # Extract data for analysis
    models = list(input.keys())
    results = {}

    # Compare normality of input for each model
    normality_results = {}
    for model in models:
        residuals = input[model]
        normality_test = test_normality(residuals)
        normality_results[model] = normality_test['conclusion']
    
    # Check if all models have normally distributed input
    normal = all(normality_results.values())

    results['Normality Test'] = {
        'all_normal': normal,
        'details': normality_results
    }

    if normal:
        # For normally distributed data, use one-way ANOVA
        # Prepare data for ANOVA
        anova_data = []
        anova_groups = []
        
        for model in models:
            anova_data.extend(input[model])
            anova_groups.extend([model] * len(input[model]))
        
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*[input[model] for model in models])
        
        # Calculate additional ANOVA details
        groups = [input[model] for model in models]
        group_means = [np.mean(group) for group in groups]
        overall_mean = np.mean(np.concatenate(groups))
        
        # Degrees of freedom
        df_between = len(groups) - 1
        df_within = sum(len(group) for group in groups) - len(groups)
        
        # Sum of squares
        ss_between = sum(len(group) * (group_mean - overall_mean)**2 for group, group_mean in zip(groups, group_means))
        ss_within = sum(sum((x - group_mean)**2 for x in group) for group, group_mean in zip(groups, group_means))
        
        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        total_ss = ss_between + ss_within
        total_df = df_between + df_within

        results['One-way ANOVA'] = {
            stat: value for stat, value in zip(
            ['f-statistic', 'p-value', 'significant', 'df_between', 'df_within', 'total_df', 'ss_between', 'ss_within', 'total_ss', 'ms_between', 'ms_within'],
            [f_statistic, p_value, p_value < alpha, df_between, df_within, total_df, ss_between, ss_within, total_ss, ms_between, ms_within]
            )
        }

        # Calculate effect sizes for ANOVA
        n_total = sum(len(group) for group in groups)
        eta_squared = ss_between / (ss_between + ss_within)
        omega_squared = (ss_between - (df_between * ms_within)) / (ss_between + ss_within + ms_within)
        epsilon_squared = (ss_between - (df_between * ms_within)) / (ss_between + ss_within)

        # Confidence intervals for eta squared
        ci_eta_lower = eta_squared - (1.96 * np.sqrt((2 * eta_squared * (1 - eta_squared)) / n_total))
        ci_eta_upper = eta_squared + (1.96 * np.sqrt((2 * eta_squared * (1 - eta_squared)) / n_total))

        # Confidence intervals for omega squared
        ci_omega_lower = omega_squared - (1.96 * np.sqrt((2 * omega_squared * (1 - omega_squared)) / n_total))
        ci_omega_upper = omega_squared + (1.96 * np.sqrt((2 * omega_squared * (1 - omega_squared)) / n_total))

        # Confidence intervals for epsilon squared
        ci_epsilon_lower = epsilon_squared - (1.96 * np.sqrt((2 * epsilon_squared * (1 - epsilon_squared)) / n_total))
        ci_epsilon_upper = epsilon_squared + (1.96 * np.sqrt((2 * epsilon_squared * (1 - epsilon_squared)) / n_total))

        results['Effect Sizes'] = {
            stat: value for stat, value in zip(
            [
                'eta_squared', 'omega_squared', 'epsilon_squared',
                'eta_CI_lower', 'eta_CI_upper',
                'omega_CI_lower', 'omega_CI_upper',
                'epsilon_CI_lower', 'epsilon_CI_upper'
            ],
            [
                eta_squared, omega_squared, epsilon_squared,
                max(0, ci_eta_lower), min(1, ci_eta_upper),
                max(0, ci_omega_lower), min(1, ci_omega_upper),
                max(0, ci_epsilon_lower), min(1, ci_epsilon_upper)
            ]
            )
        }
        
        # If ANOVA is significant, perform Tukey's HSD post-hoc test
        if p_value < alpha:
            tukey = pairwise_tukeyhsd(np.array(anova_data), np.array(anova_groups), alpha=alpha)
            results['Tukey\'s Honest Significant Difference'] = {
                'summary': {
                    'models': models,
                    'means': [np.mean(input[model]) for model in models],
                },
                'comparisons': [
                    {
                        'model_1': tukey._results_table.data[i+1][0],
                        'model_2': tukey._results_table.data[i+1][1],
                        'mean_diff': tukey.meandiffs[i],
                        'std_err': tukey.std_pairs[i],
                        'conf_int_low': tukey.confint[i, 0],
                        'conf_int_high': tukey.confint[i, 1],
                        'p-adj': tukey.pvalues[i],
                        'reject': tukey.reject[i]
                    }
                    for i in range(len(tukey.meandiffs))
                ]
            }
    else:
        # For non-normally distributed data, use Friedman test
        # Prepare data for Friedman test
        # Note: Friedman test requires equal number of samples for each model
        # We'll take the minimum common length
        min_length = min(len(input[model]) for model in models)
        
        friedman_data = np.array([input[model][:min_length] for model in models])
        
        # Perform Friedman test
        try:
            # Perform Friedman test
            chi2, p_value = friedmanchisquare(*friedman_data)
            n = friedman_data.shape[1]
            k = friedman_data.shape[0]
            df = k - 1
            
            # Calculate ranks
            ranks = np.mean(np.argsort(np.argsort(friedman_data, axis=1), axis=0) + 1, axis=1)
            
            results['Friedman Test'] = {
                stat: value for stat, value in zip(
                ['N', 'chi2', 'df', 'asymp_sig', 'significant'],
                [n, chi2, df, p_value, p_value < alpha]
                )
            }
            
            # If Friedman test is significant, perform post-hoc test (Nemenyi)
            if p_value < alpha:
                posthoc = sp.posthoc_nemenyi_friedman(friedman_data.T)
                
                # Calculate additional details for Nemenyi test
                n = friedman_data.shape[1]
                k = friedman_data.shape[0]
                ranks = np.mean(np.argsort(np.argsort(friedman_data, axis=1), axis=0) + 1, axis=1)
                r_sum = np.sum(np.argsort(np.argsort(friedman_data, axis=1), axis=0) + 1, axis=1)
                r_mean = ranks
                
                nemenyi_results = {
                    'models': list(models),
                    'size': [n]*k,
                    'r_sum': r_sum.tolist(),
                    'r_mean': r_mean.tolist()
                }
                
                # Pairwise comparisons
                comparisons = []
                for i in range(k):
                    for j in range(i + 1, k):
                        mean_diff = abs(r_mean[i] - r_mean[j])
                        p_value = posthoc.iloc[i, j]
                        comparisons.append({
                            'model_1': list(models)[i],
                            'model_2': list(models)[j],
                            'r_mean_diff': mean_diff,
                            'p-value': p_value,
                            'significant': p_value < alpha
                        })
                
                results['Nemenyi Test'] = {
                    'summary': nemenyi_results,
                    'comparisons': comparisons
                }
        except Exception as e:
            results['error'] = str(e)
    
    return results

def create_dataframes_from_stats(json_data):
    """Create DataFrames from statistical test results."""
    dataframes = {}
    
    for key, value in json_data.items():
        if key == "One-way ANOVA" or key == "Effect Sizes" or key == "Friedman Test":
            # Simple key-value sections to DataFrame
            df = pd.DataFrame({'stats': list(value.keys()), 'values': list(value.values())})
            dataframes[key] = df
        
        if key == "Normality Test":
            # Simple key-value sections to DataFrame
            df = pd.DataFrame({'stats': list(value['details'].keys()), 'values': list(value['details'].keys())})
            dataframes[key] = df

        elif key == "Tukey's Honest Significant Difference":
            # Extract comparison data
            if "comparisons" in value:
                df = pd.DataFrame(value["comparisons"])
                dataframes[key] = df
        
        elif key == "Nemenyi Test":
            # Handle both the comparisons and summary
            if "comparisons" in value:
                comparisons_df = pd.DataFrame(value["comparisons"])
                dataframes[f"{key}_comparisons"] = comparisons_df
            
            if "summary" in value:
                summary = value["summary"]
                model_ranks_df = pd.DataFrame(summary)
                dataframes[f"{key}_summary"] = model_ranks_df
    
    return dataframes

def process_stats_files(results_path, combinations):
    """
    Process statistical test result JSON files and save as CSV files.
    
    Args:
        results_path: Base path to the results directory
        combinations: List of (hc, rc) tuples representing combinations to process
    """
    for hc, rc in combinations:
        # Construct the file path
        file_path = f"{results_path}/{hc}/{rc}/{hc}_{rc}_stats_test_result.json"
        print(f"Processing: {file_path}")
        
        try:
            # Ensure the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            # Load JSON file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create DataFrames
            dataframes = create_dataframes_from_stats(json.loads(content))

            # Save each DataFrame to CSV
            for key, df in dataframes.items():
                output_filename = f"{results_path}/{hc}/{rc}/{hc}_{rc}_{key}.csv"
                df.to_csv(output_filename, index=False)
                print(f"Saved: {output_filename}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")