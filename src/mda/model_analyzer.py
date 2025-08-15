import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
from statsmodels.stats.libqsturng import qsturng
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import f_oneway

from mda.stats_testing import test_statistical_significance

class ModelPerformanceAnalyzer:
    def __init__(self, alpha=0.05):
        """
        Initialize the model performance analyzer.
        
        Parameters:
        -----------
        alpha : float, default=0.05
            The significance level for statistical tests.
        """
        self.alpha = alpha
        self.experiment_results = {}
        self.summary_results = {}
        
    def add_experiment(self, experiment_name, sample):
        """
        Add experiment results to the analyzer.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment.
        sample : dict
            Dictionary with model names as keys and lists of sample as values.
            All models should have the same number of samples.
        """
        # Check if all models have the same number of samples
        lengths = [len(v) for v in sample.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All models must have the same number of samples")
            
        # Store the results
        self.experiment_results[experiment_name] = sample
        
    def analyze_experiment(self, experiment_name):
        """
        Perform statistical analysis on a single experiment.
        First tests for normality to determine whether to use parametric or non-parametric tests.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment to analyze.
            
        Returns:
        --------
        dict
            Results of the statistical analysis.
        """
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment '{experiment_name}' not found")
            
        sample = self.experiment_results[experiment_name]
        
        return test_statistical_significance(sample)
    
    def analyze_all_experiments(self):
        """
        Perform statistical analysis on all experiments.
        
        Returns:
        --------
        dict
            Results of all statistical analyses.
        """
        all_results = {}
        
        for experiment_name in self.experiment_results:
            all_results[experiment_name] = self.analyze_experiment(experiment_name)
            
        return all_results
    
    def summarize_performance(self):
        """
        Create a summary of model performance across all experiments.
        
        Returns:
        --------
        dict
            Summary of model performance.
        """
        # First, analyze all experiments if not already done
        all_results = self.analyze_all_experiments()
        
        # Extract all model names across all experiments
        all_models = set()
        for experiment in self.experiment_results.values():
            all_models.update(experiment.keys())
        all_models = list(all_models)
        
        # Initialize summary structures
        win_matrix = pd.DataFrame(0, index=all_models, columns=all_models)
        tie_matrix = pd.DataFrame(0, index=all_models, columns=all_models)
        loss_matrix = pd.DataFrame(0, index=all_models, columns=all_models)
        
        # Track average ranks
        rank_sum = defaultdict(float)
        rank_count = defaultdict(int)
        
        # Track overall win/tie/loss
        overall_wins = defaultdict(int)
        overall_ties = defaultdict(int)
        overall_losses = defaultdict(int)
        
        # Populate matrices
        for exp_name, results in all_results.items():
            # Check if we used parametric or non-parametric tests
            if 'Tukey\'s Honest Significant Difference' in results:
                tukey = results['Tukey\'s Honest Significant Difference']
                
                # Record means (lower is better for error metrics)
                for i, model in enumerate(tukey['summary']['models']):
                    # Use negative means because lower is better for error metrics
                    # This makes ranking consistent with higher = better
                    rank_sum[model] += -tukey['summary']['means'][i]
                    rank_count[model] += 1
                
                # Record pairwise comparisons
                for comp in tukey['comparisons']:
                    model_1 = comp['model_1']
                    model_2 = comp['model_2']
                    
                    # Use significance flag for determination
                    if comp['reject']:
                        # Mean diff is model_1 - model_2, so negative means model_1 is better (lower error)
                        if comp['mean_diff'] < 0:
                            win_matrix.loc[model_1, model_2] += 1
                            loss_matrix.loc[model_2, model_1] += 1
                            overall_wins[model_1] += 1
                            overall_losses[model_2] += 1
                        else:
                            win_matrix.loc[model_2, model_1] += 1
                            loss_matrix.loc[model_1, model_2] += 1
                            overall_wins[model_2] += 1
                            overall_losses[model_1] += 1
                    else:
                        # No significant difference
                        tie_matrix.loc[model_1, model_2] += 1
                        tie_matrix.loc[model_2, model_1] += 1
                        overall_ties[model_1] += 1
                        overall_ties[model_2] += 1
            
            elif 'Nemenyi Test' in results:
                nemenyi = results['Nemenyi Test']
                
                # Record ranks
                for i, model in enumerate(nemenyi['summary']['models']):
                    rank_sum[model] += nemenyi['summary']['r_mean'][i]
                    rank_count[model] += 1
                
                # Record pairwise comparisons
                for comp in nemenyi['comparisons']:
                    model_1 = comp['model_1']
                    model_2 = comp['model_2']
                    
                    # Use p-value for significance determination
                    if comp['significant']:
                        if comp['r_mean_diff'] > 0:  # Check which model is better
                            # Lower rank means better performance
                            idx_1 = nemenyi['summary']['models'].index(model_1)
                            idx_2 = nemenyi['summary']['models'].index(model_2)
                            
                            if nemenyi['summary']['r_mean'][idx_1] > nemenyi['summary']['r_mean'][idx_2]:
                                win_matrix.loc[model_1, model_2] += 1
                                loss_matrix.loc[model_2, model_1] += 1
                                overall_wins[model_1] += 1
                                overall_losses[model_2] += 1
                            else:
                                win_matrix.loc[model_2, model_1] += 1
                                loss_matrix.loc[model_1, model_2] += 1
                                overall_wins[model_2] += 1
                                overall_losses[model_1] += 1
                    else:
                        # No significant difference
                        tie_matrix.loc[model_1, model_2] += 1
                        tie_matrix.loc[model_2, model_1] += 1
                        overall_ties[model_1] += 1
                        overall_ties[model_2] += 1
        
        # Calculate average ranks
        avg_ranks = {model: rank_sum[model] / rank_count[model] if rank_count[model] > 0 else float('nan') 
                     for model in all_models}
        
        # Sort models by average rank
        sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        # Summarize relative performance
        relative_performance = {}
        for model in all_models:
            relative_performance[model] = {
                'wins': overall_wins[model],
                'ties': overall_ties[model] // 2,  # Divide by 2 because ties are counted twice
                'losses': overall_losses[model],
                'avg_rank': avg_ranks[model]
            }
        
        # Create final summary
        summary = {
            'win_matrix': win_matrix.to_dict(),
            'tie_matrix': tie_matrix.to_dict(),
            'loss_matrix': loss_matrix.to_dict(),
            'avg_ranks': avg_ranks,
            'relative_performance': relative_performance,
            'best_model': sorted_models[0][0] if sorted_models else None
        }
        
        self.summary_results = summary
        return summary
    
    def generate_results_table(self):
        """
        Generate a formatted results table.
        
        Returns:
        --------
        pandas.DataFrame
            Table of model performance across experiments.
        """
        if not self.summary_results:
            self.summarize_performance()
            
        # Create DataFrame for relative performance
        perf_data = []
        for model, stats in self.summary_results['relative_performance'].items():
            perf_data.append({
                'Model': model,
                'Wins': stats['wins'],
                'Ties': stats['ties'],
                'Losses': stats['losses'],
                'Average Rank': round(stats['avg_rank'], 2)
            })
            
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.sort_values(by='Average Rank')
        
        return perf_df
    
    def plot_rank_summary(self):
        """
        Create a visualization of average ranks across experiments.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with rank visualization.
        """
        if not self.summary_results:
            self.summarize_performance()
            
        # Create DataFrame for ranks
        rank_data = pd.DataFrame({
            'Model': list(self.summary_results['avg_ranks'].keys()),
            'Average Rank': list(self.summary_results['avg_ranks'].values())
        })
        
        rank_data = rank_data.sort_values(by='Average Rank')
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Average Rank', data=rank_data)
        ax.set_title('Average Ranks Across All Experiments')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Rank (lower is better)')
        
        # Add values on top of bars
        for i, v in enumerate(rank_data['Average Rank']):
            ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_win_loss_matrix(self):
        """
        Create a heatmap visualization of the win/loss matrix.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with win/loss heatmap.
        """
        if not self.summary_results:
            self.summarize_performance()
            
        # Convert win_matrix back to DataFrame if it's stored as dict
        if isinstance(self.summary_results['win_matrix'], dict):
            win_matrix = pd.DataFrame(self.summary_results['win_matrix'])
        else:
            win_matrix = self.summary_results['win_matrix']
            
        # Create plot
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(win_matrix, annot=True, cmap='YlGnBu', fmt='d')
        ax.set_title('Win Matrix (Row Model vs Column Model)')
        plt.tight_layout()
        return plt.gcf()
        
    def generate_detailed_report(self):
        """
        Generate a comprehensive report of the analysis.
        
        Returns:
        --------
        str
            Detailed report in formatted text.
        """
        if not self.summary_results:
            self.summarize_performance()
            
        # Start building the report
        report = []
        report.append("# Model Performance Analysis Report")
        report.append("\n## Summary of Results")
        
        # Add best model
        best_model = self.summary_results['best_model']
        report.append(f"\nBest performing model: **{best_model}**")
        
        # Add table of average ranks
        report.append("\n### Average Ranks (lower is better)")
        rank_items = sorted(self.summary_results['avg_ranks'].items(), key=lambda x: x[1])
        report.append("\n| Model | Average Rank |")
        report.append("| ----- | ------------ |")
        for model, rank in rank_items:
            report.append(f"| {model} | {rank:.2f} |")
        
        # Add table of wins/ties/losses
        report.append("\n### Win/Tie/Loss Summary")
        report.append("\n| Model | Wins | Ties | Losses |")
        report.append("| ----- | ---- | ---- | ------ |")
        for model, rank in rank_items:  # Use same order as ranks
            stats = self.summary_results['relative_performance'][model]
            report.append(f"| {model} | {stats['wins']} | {stats['ties']} | {stats['losses']} |")
        
        # Add experiment-specific results
        report.append("\n## Detailed Results by Experiment")
        all_results = self.analyze_all_experiments()
        
        for exp_name, results in all_results.items():
            report.append(f"\n### Experiment: {exp_name}")
            
            # Normality test results
            normality = results['Normality Test']
            report.append(f"\nNormality test conclusion: {'All distributions are normal' if normality['all_normal'] else 'Non-normal distributions detected'}")
            
            # Show which test was used based on normality
            if normality['all_normal']:
                # ANOVA test results
                anova = results['One-way ANOVA']
                report.append(f"\nANOVA Test: statistic = {anova['f-statistic']:.4f}, p-value = {anova['p-value']:.4f}")
                report.append(f"Significance: {'Significant' if anova['significant'] else 'Not significant'}")
                
                # Tukey's HSD results (if applicable)
                if 'Tukey\'s Honest Significant Difference' in results:
                    tukey = results['Tukey\'s Honest Significant Difference']
                    report.append("\n#### Tukey's HSD Summary")
                    
                    # Add table of means
                    report.append("\n| Model | Mean Error |")
                    report.append("| ----- | ---------- |")
                    for i, model in enumerate(tukey['summary']['models']):
                        report.append(f"| {model} | {tukey['summary']['means'][i]:.4f} |")
                    
                    # Add table of pairwise comparisons
                    report.append("\n#### Pairwise Comparisons")
                    report.append("\n| Model 1 | Model 2 | Mean Diff | p-value | 95% CI | Significant? |")
                    report.append("| ------- | ------- | --------- | ------- | ------ | ------------ |")
                    
                    for comp in tukey['comparisons']:
                        sig_str = "Yes" if comp['reject'] else "No"
                        ci_str = f"[{comp['lower']:.4f}, {comp['upper']:.4f}]"
                        report.append(f"| {comp['model_1']} | {comp['model_2']} | " +
                                     f"{comp['mean_diff']:.4f} | {comp['p-adj']:.4f} | {ci_str} | {sig_str}")
            else:
                # Friedman test results
                friedman = results['Friedman Test']
                report.append(f"\nFriedman Test: statistic = {friedman['chi2']:.4f}, p-value = {friedman['chi2']:.4f}")
                report.append(f"Significance: {'Significant' if friedman['significant'] else 'Not significant'}")
                
                # Nemenyi test results (if applicable)
                if 'Nemenyi Test' in results:
                    nemenyi = results['Nemenyi Test']
                    report.append("\n#### Nemenyi Test Summary")
                    report.append(f"\nNumber of samples: {nemenyi['summary']['size']}")
                    
                    # Add table of mean ranks
                    report.append("\n| Model | Mean Rank |")
                    report.append("| ----- | --------- |")
                    for i, model in enumerate(nemenyi['summary']['models']):
                        report.append(f"| {model} | {nemenyi['summary']['r_mean'][i]:.2f} |")
                    
                    # Add table of pairwise comparisons
                    report.append("\n#### Pairwise Comparisons")
                    report.append("\n| Model 1 | Model 2 | Mean Diff | p-value | Significant? |")
                    report.append("| ------- | ------- | --------- | ------- | ------------ |")
                    
                    for comp in nemenyi['comparisons']:
                        sig_str = "Yes" if comp['significant'] else "No"
                        report.append(f"| {comp['model_1']} | {comp['model_2']} | " +
                                    f"{comp['r_mean_diff']:.4f} | {comp['p-value']:.4f} | {sig_str} |")
        
        return "\n".join(report)