import numpy as np
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import qsturng
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

import warnings
warnings.filterwarnings('ignore')

def test_normality(residuals, alpha=0.05):
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
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # D'Agostino's K^2 test
    dagostino_stat, dagostino_p = stats.normaltest(residuals)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(residuals, 'norm')
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

def test_homoscedasticity(target, predictors, residuals, alpha=0.05):
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
    gq_stat, gq_p, _ = het_goldfeldquandt(target, X)
    
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

def test_statistical_significance(errors, alpha=0.05):
    normal = True
    # Compare normality of errors for each model
    normality_results = {}
    for model in errors.keys():
        residuals = np.concatenate(list(errors[model].values()))
        normality_test = test_normality(residuals)
        normality_results[model] = normality_test['conclusion']
    
    # Check if all models have normally distributed errors
    normal = all(normality_results.values())

    results = {}
    if normal:
        # For normally distributed data, use one-way ANOVA
        # Prepare data for ANOVA
        anova_data = []
        anova_groups = []
        
        for model in errors.keys():
            anova_data.extend(np.concatenate(list(errors[model].values())))
            anova_groups.extend([model] * len(np.concatenate(list(errors[model].values()))))
        
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*[np.concatenate(list(errors[model].values())) for model in errors.keys()])
        
        # Calculate additional ANOVA details
        groups = [np.concatenate(list(errors[model].values())) for model in errors.keys()]
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
        
        results['One-way ANOVA'] = {
            'f_statistic': f_statistic,
            'p-value': p_value,
            'significant': p_value < alpha,
            'df_between': df_between,
            'df_within': df_within,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ms_between': ms_between,
            'ms_within': ms_within
        }
        # Calculate effect sizes for ANOVA
        n_total = sum(len(group) for group in groups)
        eta_squared = ss_between / (ss_between + ss_within)
        omega_squared = (ss_between - (df_between * ms_within)) / (ss_between + ss_within + ms_within)
        epsilon_squared = (ss_between - (df_between * ms_within)) / (ss_between + ss_within)

        # Confidence intervals for eta squared
        ci_lower = eta_squared - (1.96 * np.sqrt((2 * eta_squared * (1 - eta_squared)) / n_total))
        ci_upper = eta_squared + (1.96 * np.sqrt((2 * eta_squared * (1 - eta_squared)) / n_total))

        results['Effect Sizes'] = {
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'epsilon_squared': epsilon_squared,
            'CI_lower': max(0, ci_lower),
            'CI_upper': min(1, ci_upper),
        }
        
        # If ANOVA is significant, perform Tukey's HSD post-hoc test
        if p_value < alpha:
            tukey = pairwise_tukeyhsd(np.array(anova_data), np.array(anova_groups), alpha=alpha)
            results['Tukey\'s Honest Significant Difference'] = {
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
        min_length = min(len(np.concatenate(list(errors[model].values()))) for model in errors.keys())
        
        friedman_data = np.array([np.concatenate(list(errors[model].values()))[:min_length] for model in errors.keys()])
        
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
                'N': n,
                'chi2': chi2,
                'df': df,
                'asymp_sig': p_value,
                'significant': p_value < alpha,
                'ranks': ranks.tolist()
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
                q_crit = qsturng(1 - alpha, k, np.inf)
                
                nemenyi_results = {
                    'models': list(errors.keys()),
                    'r_sum': r_sum.tolist(),
                    'size': n,
                    'r_mean': r_mean.tolist(),
                    'q_crit': q_crit
                }
                
                # Pairwise comparisons
                comparisons = []
                for i in range(k):
                    for j in range(i + 1, k):
                        mean_diff = abs(r_mean[i] - r_mean[j])
                        std_error = np.sqrt(k * (k + 1) / (6 * n))
                        q_stat = mean_diff / std_error
                        r_crit = std_error*q_crit
                        p_value = posthoc.iloc[i, j]
                        comparisons.append({
                            'model_1': list(errors.keys())[i],
                            'model_2': list(errors.keys())[j],
                            'r_mean_diff': mean_diff,
                            'std_error': std_error,
                            'q_stat': q_stat,
                            'p-value': p_value,
                            'r_crit': r_crit,
                            'p_significant': p_value < alpha,
                            'q_significant': q_stat > q_crit,
                            'r_significant': mean_diff > r_crit
                        })
                
                results['Nemenyi Test'] = {
                    'summary': nemenyi_results,
                    'comparisons': comparisons
                }
        except Exception as e:
            results['error'] = str(e)
    
    return results