import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

from models import CalciferNet
from mda.stats_testing import test_normality, test_homoscedasticity, test_statistical_significance

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):  # Convert numpy.bool_ to Python bool
            return bool(obj)
        if isinstance(obj, np.integer):  # Convert numpy integers to Python int
            return int(obj)
        if isinstance(obj, np.floating):  # Convert numpy floats to Python float
            return float(obj)
        if isinstance(obj, np.ndarray):  # Convert numpy arrays to lists
            return obj.tolist()
        return super().default(obj)
    
class ModelManager:
    def __init__(self):
        """Initialize an empty model manager to store and manage CalciferNet models."""
        self.models = {}
        self.model_groups = {}
    
    def add_model(self, model_name, model_instance):
        """
        Add a model to the manager.
        
        Parameters:
            model_name (str): Unique identifier for the model
            model_instance (CalciferNet): Instance of the CalciferNet model
            group (str, optional): Group name to categorize the model
        """
        if model_name in self.models:
            print(f"Warning: Overwriting existing model '{model_name}'")
        
        self.models[model_name] = model_instance
        
    def add_to_group(self, model_name, group=None):
        if group:
            if group not in self.model_groups:
                self.model_groups[group] = []
            self.model_groups[group].append(model_name)
    
    def create_model(self, model_name, model_type, groups=None):
        """
        Create a new CalciferNet model and add it to the manager.
        
        Parameters:
            model_name (str): Unique identifier for the model
            model_type (str): Type of the model
            model_file (str): Filename of the model to load
            X_scaler_file (str): Filename of the feature scaler
            y_scaler_file (str): Filename of the target scaler
            group (str, optional): Group name to categorize the model
            
        Returns:
            CalciferNet: The created model instance
        """
        model = CalciferNet(model_name, *model_type)

        self.add_model(model_name, model)
        if isinstance(groups, list):
            for group in groups:
                self.add_to_group(model_name, group)
        else:
            self.add_to_group(model_name, groups)

        return model
    
    def get_model(self, model_name):
        """
        Retrieve a model by name.
        
        Parameters:
            model_name (str): Name of the model to retrieve
            
        Returns:
            CalciferNet: The requested model instance
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found in the manager")
        
        return self.models[model_name]
    
    def get_models_by_group(self, group):
        """
        Retrieve all models in a specific group.
        
        Parameters:
            group (str): Group name
            
        Returns:
            dict: Dictionary of model_name -> model_instance for all models in the group
        """
        if group not in self.model_groups:
            raise KeyError(f"Group '{group}' not found in the manager")
        
        return {name: self.models[name] for name in self.model_groups[group]}
    
    def list_models(self):
        """
        List all models in the manager.
        
        Returns:
            dict: Dictionary with model names as keys and model types as values
        """
        return {name: name for name, model in self.models.items()}
    
    def list_groups(self):
        """
        List all groups and their models.
        
        Returns:
            dict: Dictionary with group names as keys and lists of model names as values
        """
        return self.model_groups.copy()
    
    def batch_predict(self, data, target_col='Tc', model_names=None, group=None):
        """
        Run predictions on multiple models at once.
        
        Parameters:
            input_data (pd.DataFrame): Input data for prediction
            model_names (list, optional): List of model names to use for prediction
            group (str, optional): Group name to use for prediction
            
        Returns:
            dict: Dictionary with model names as keys and prediction results as values
        """
        results = {}
        
        # Determine which models to use
        models_to_use = {}
        if model_names:
            models_to_use = {name: self.get_model(name) for name in model_names}
        elif group:
            models_to_use = self.get_models_by_group(group)
        else:
            models_to_use = self.models
        
        # Run predictions
        for name, model in models_to_use.items():
            if 'ensemble' not in model.name:
                results[name] = self.auto_predict(data, target_col, name)
            else:
                results[name] = model.predict(data)
        
        return results

    def evaluate_models(self, data, target_col='Tc', model_names=None, group=None):
        """
        Evaluate multiple models against a dataset with known targets.
        
        Parameters:
            data (pd.DataFrame): Dataset containing features and true target values
            target_col (str): Column name for the true target values
            model_names (list, optional): List of model names to evaluate
            group (str, optional): Group name to evaluate
            
        Returns:
            dict: Dictionary with model names as keys and evaluation metrics as values
        """
        def calculate_metrics(y_true, y_pred):
            metrics = {
                'RMSE': root_mean_squared_error(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'R²': r2_score(y_true, y_pred)
            }

            return metrics
        
        evaluation_results = {}
        
        # Determine which models to use
        models_to_use = {}
        if model_names:
            models_to_use = {name: self.get_model(name) for name in model_names}
        elif group:
            models_to_use = self.get_models_by_group(group)
        else:
            models_to_use = self.models
        
        # Run predictions and calculate metrics
        for name, model in models_to_use.items():
            # Make predictions
            if 'ensemble' not in model.name:
                predictions = self.auto_predict(data, target_col, name)
            else:
                predictions = model.predict(data, 'Tc')
            
            # Calculate metrics
            y_true = data[target_col].values
            if isinstance(predictions, pd.DataFrame):
                y_pred = predictions.values
            else:
                y_pred = predictions
            
            # Reshape if needed (depends on your predict method's output)
            if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.flatten()
            
            metrics = calculate_metrics(y_true, y_pred)
            evaluation_results[name] = metrics
        
        return evaluation_results
    
    def compare_models(self, data, conditions, target_col='Tc', model_names=None, group=None, to_sort=False, plot=True):
        """
        Compare multiple models against a dataset with known targets.
        
        Parameters:
            data (pd.DataFrame): Dataset containing features and true target values
            target_col (str): Column name for the true target values
            model_names (list, optional): List of model names to compare
            group (str, optional): Group name to compare
            sorted (bool): Whether to sort results by RMSE
            plot (bool): Whether to generate comparison plots
            
        Returns:
            tuple: (metrics_df, error_stats_df) - DataFrames with comparison metrics and error statistics
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # Initialize an empty DataFrame to store comparison results
        comparison_df = pd.DataFrame()

        # Group data by 'trial' and evaluate models for each trial
        for trial, trial_data in data.groupby('trial'):
            # Get evaluation results for all specified models for the current trial
            eval_results = self.evaluate_models(trial_data.copy(), target_col, model_names, group)
            
            # Convert to DataFrame and add a 'trial' column for easier comparison
            trial_comparison_df = pd.DataFrame.from_dict(eval_results, orient='index').reset_index().rename(columns={'index': 'Model'})
            trial_comparison_df['trial'] = trial
            
            # Append the results for the current trial to the main DataFrame
            comparison_df = pd.concat([comparison_df, trial_comparison_df], axis=0)

        if to_sort:
            comparison_df = comparison_df.sort_values(['trial', 'RMSE'])
        
        # Display the comparison in a well-formatted way
        for trial in comparison_df['trial'].unique():
            print(f"\n---------------\n Trial {trial} \n---------------")
            trial_metrics = comparison_df[comparison_df['trial'] == trial]
            print(trial_metrics.drop(columns=['trial']).to_string(index=False))
        
        # Determine which models to use
        models_to_use = {}
        if model_names:
            models_to_use = {name: self.get_model(name) for name in model_names}
        elif group:
            models_to_use = self.get_models_by_group(group)
        else:
            models_to_use = self.models
        
        # Create a DataFrame with predictions from all models
        results_df = data.copy()
        errors = {}
        for name, model in models_to_use.items():
            features = data[['Tg', 'Ta']]
            
            # Make predictions and add to results DataFrame
            predictions = model.predict(features)
            if isinstance(predictions, pd.DataFrame):
                predictions = predictions.to_numpy().round(1)
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten().round(1)
            
            results_df[f'{name}_pred'] = predictions
            # Calculate errors
            results_df[f'{name}_error'] = results_df[target_col] - results_df[f'{name}_pred']
            errors[name] = {
                trial: results_df[results_df['trial'] == trial][f'{name}_error'].values
                for trial in results_df['trial'].unique()
            }

        # Create error statistics DataFrame
        error_stats = {}
        over_all_error_stats = {}
        normality_stats = []
        homoscedasticity_stats = []
        assumption_result = {}
        for name in models_to_use.keys():
            error_stats[name] = {
                trial: {
                    'N': len(errors[name][trial]),
                    'Mean': np.mean(errors[name][trial]),
                    'Std Dev': np.std(errors[name][trial]),
                    'Standard Error': stats.sem(errors[name][trial]),
                    'CI Lower': stats.t.interval(
                        0.95,  # Confidence level (95%)
                        len(errors[name][trial]) - 1,  # Degrees of freedom
                        loc=np.mean(errors[name][trial]),  # Mean
                        scale=stats.sem(errors[name][trial])  # Standard error
                    )[0],  # Lower bound
                    'CI Upper': stats.t.interval(
                        0.95,  # Confidence level (95%)
                        len(errors[name][trial]) - 1,  # Degrees of freedom
                        loc=np.mean(errors[name][trial]),  # Mean
                        scale=stats.sem(errors[name][trial])  # Standard error
                    )[1],  # Upper bound
                    'Min': np.min(errors[name][trial]),
                    'Max': np.max(errors[name][trial]),
                    'Range': np.max(errors[name][trial]) - np.min(errors[name][trial]),
                    'IQR': np.percentile(errors[name][trial], 75) - np.percentile(errors[name][trial], 25),
                    'Skewness': stats.skew(errors[name][trial]),
                    'Kurtosis': stats.kurtosis(errors[name][trial])
                }
                for trial in errors[name].keys()
            }

            # Calculate overall error statistics
            overall_errors = np.concatenate(list(errors[name].values()))
            over_all_error_stats[name] = {
                'N': len(overall_errors),
                'Mean': np.mean(overall_errors),
                'Std Dev': np.std(overall_errors),
                'Standard Error': stats.sem(overall_errors),
                'CI Lower': stats.t.interval(
                    0.95,  # Confidence level (95%)
                    len(overall_errors) - 1,  # Degrees of freedom
                    loc=np.mean(overall_errors),  # Mean
                    scale=stats.sem(overall_errors)  # Standard error
                )[0],  # Lower bound
                'CI Upper': stats.t.interval(
                    0.95,  # Confidence level (95%)
                    len(overall_errors) - 1,  # Degrees of freedom
                    loc=np.mean(overall_errors),  # Mean
                    scale=stats.sem(overall_errors)  # Standard error
                )[1],
                'Min': np.min(overall_errors),
                'Max': np.max(overall_errors),
                'Range': np.max(overall_errors) - np.min(overall_errors),
                'IQR': np.percentile(overall_errors, 75) - np.percentile(overall_errors, 25),
                'Skewness': stats.skew(overall_errors),
                'Kurtosis': stats.kurtosis(overall_errors)
            }
        
            for trial in results_df['trial'].unique():
                trial_data = results_df[results_df['trial'] == trial]
                normality = test_normality(trial_data[f'{name}_error'])
                homoscedasticity = test_homoscedasticity(
                    trial_data['Tc'], 
                    trial_data[['Tg', 'Ta']], 
                    trial_data[f'{name}_error']
                )
                
                assumption_result[(name, trial)] = {
                    'normality': normality['conclusion'],
                    'homoscedasticity': homoscedasticity['conclusion']
                }

                normality.pop('conclusion')
                normality_stats += [
                    {
                        'model': name,
                        'trial': trial,
                        'test': test_name,
                        **test_results
                    }
                    for test_name, test_results in normality.items()
                ]

                homoscedasticity.pop('conclusion')
                homoscedasticity_stats += [
                    {
                        'model': name,
                        'trial': trial,
                        'test': test_name,
                        **test_results
                    }
                    for test_name, test_results in homoscedasticity.items()
                ]

        stats_test_result = test_statistical_significance(errors)

        error_stats_flat = [
            {'Model': name, 'Trial': trial, **stats}
            for name, trials in error_stats.items()
            for trial, stats in trials.items()
        ]
        error_stats_df = pd.DataFrame(error_stats_flat)
        overall_error_stats_df = pd.DataFrame.from_dict(over_all_error_stats, orient='index').reset_index().rename(columns={'index': 'Model'})
        normality_stats_df = pd.DataFrame(normality_stats)
        homoscedasticity_stats_df = pd.DataFrame(homoscedasticity_stats)
        assumption_result_df = pd.DataFrame.from_dict(assumption_result, orient='index').reset_index().rename(columns={'level_0': 'model', 'level_1': 'trial'})

        # Save DataFrames to CSV files
        data_path = f'../reports/results/testing/{conditions[0]}/{conditions[1]}'
        os.makedirs(data_path, exist_ok=True)
        comparison_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_comparison.csv', index=False)
        error_stats_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_error_stats.csv', index=False)
        overall_error_stats_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_overall_error_stats.csv', index=False)
        normality_stats_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_normality_stats.csv', index=False)
        homoscedasticity_stats_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_homoscedasticity_stats.csv', index=False)
        assumption_result_df.to_csv(f'{data_path}/{conditions[0]}_{conditions[1]}_assumption_results.csv', index=False)

        # Save statistical test results to a JSON file
        with open(f'{data_path}/{conditions[0]}_{conditions[1]}_stats_test_result.json', 'w') as f:
            json.dump(stats_test_result, f, indent=4, cls=NumpyEncoder)
            
        # Display the error statistics in a well-formatted way
        for trial in error_stats_df['Trial'].unique():
            print(f"\n---------------\n Trial {trial} \n---------------")
            trial_error_stats = error_stats_df[error_stats_df['Trial'] == trial]
            print(trial_error_stats.drop(columns=['Trial']).to_string(index=False))

        # Display the normality results by trial
        print("\n Normality by Trial:")
        for trial in normality_stats_df['trial'].unique():
            print(f"\nTrial {trial}:")
            trial_normality = normality_stats_df[normality_stats_df['trial'] == trial]
            print(trial_normality.to_string(index=False))

        # Display the homoscedasticity results by trial
        print("\n Homoscedasticity by Trial:")
        for trial in homoscedasticity_stats_df['trial'].unique():
            print(f"\nTrial {trial}:")
            trial_homoscedasticity = homoscedasticity_stats_df[homoscedasticity_stats_df['trial'] == trial]
            print(trial_homoscedasticity.to_string(index=False))

        # Display the assumption results by trial
        print("\n Assumptions by Trial:")
        for trial in assumption_result_df['trial'].unique():
            print(f"\nTrial {trial}:")
            trial_assumptions = assumption_result_df[assumption_result_df['trial'] == trial]
            print(trial_assumptions.to_string(index=False))
        
        # Display statistical test results in an organized manner
        print("\n Statistical Test Results:")
        print(stats_test_result)
        
        # Plot if requested
        if plot:
            figure_path = f'../reports/figures/testing/{conditions[0]}/{conditions[1]}'
            os.makedirs(figure_path, exist_ok=True)
            # Set the style for all plots
            sns.set_style("whitegrid")
            plt.rcParams.update({'font.size': 12})
            
            # 1. Plot predicted vs glabella temperatures for all models
            plt.figure(figsize=(12, 8))
            plt.scatter(results_df['Tg'], results_df[target_col], alpha=0.5, label='Actual', color='black')
            
            for name in models_to_use.keys():
                plt.scatter(results_df['Tg'], results_df[f'{name}_pred'], alpha=0.5, label=name)
            
            plt.xlabel("Glabella Temperature")
            plt.ylabel("Actual & Predicted Core Temperature")
            plt.title("Glabella Temperature vs Actual and Predicted Core Temperature - Model Comparison")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{figure_path}/{conditions[0]}_{conditions[1]}_glabella_vs_act&pred.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 2. Figure pairing scatter plots with LOESS trend lines and Q-Q plots (2 models per row)
            model_names_list = list(models_to_use.keys())
            num_models = len(model_names_list)
            
            # Calculate number of rows needed
            rows = int(np.ceil(num_models / 2))
            
            fig = plt.figure(figsize=(20, 5 * rows))
            
            for i, name in enumerate(model_names_list):
                errors = results_df[f'{name}_error']
                predictions = results_df[f'{name}_pred']
                
                # Scatter plot with LOESS trend line
                ax1 = fig.add_subplot(rows, 4, i*2+1)
                ax1.scatter(predictions, errors, alpha=0.5, color=f'C{i}')
                
                # Compute LOESS trend line
                loess_result = lowess(errors, predictions, frac=0.3)
                ax1.plot(loess_result[:, 0], loess_result[:, 1], 'r-', linewidth=2)
                
                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
                ax1.set_xlabel("Predicted Values")
                ax1.set_ylabel("Error")
                ax1.set_title(f"{name} - Error vs Predicted")
                ax1.grid(True, alpha=0.3)
                
                # Q-Q plot
                ax2 = fig.add_subplot(rows, 4, i*2+2)
                stats.probplot(errors, plot=ax2)
                ax2.set_title(f"{name} - Q-Q Plot")
            
            plt.tight_layout()
            plt.savefig(f'{figure_path}/{conditions[0]}_{conditions[1]}_scatter_qq.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 3. Box plots of errors for all models
            error_data = pd.DataFrame()

            for name in models_to_use.keys():
                error_data[name] = results_df[f'{name}_error']

            # Create a figure with a gridspec that has 3 rows (2 for distribution plots, 1 for boxplot)
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])

            # Create the 2x4 grid of distribution plots
            model_names = list(models_to_use.keys())
            for i in range(8):
                row = i // 4  # 0 for first row, 1 for second row
                col = i % 4   # 0-3 for columns
                
                if i < len(model_names):  # Make sure we don't exceed the number of models
                    ax = fig.add_subplot(gs[row, col])
                    name = model_names[i]
                    sns.histplot(error_data[name], kde=True, ax=ax)
                    ax.set_title(f"{name}\nError Distribution")
                    ax.set_xlabel("Error")

            # Create the consolidated box plot at the bottom spanning all columns
            ax_box = fig.add_subplot(gs[2, :])
            sns.boxplot(data=error_data, ax=ax_box)
            ax_box.set_title("Error Box Plots by Model")
            ax_box.set_ylabel("Error")
            ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45)

            plt.tight_layout()
            plt.savefig(f'{figure_path}/{conditions[0]}_{conditions[1]}_error_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return comparison_df, \
                error_stats_df, \
                overall_error_stats_df, \
                normality_stats_df, \
                homoscedasticity_stats_df, \
                assumption_result_df, \
                stats_test_result
    
    # Add this to enable automatic feature selection during prediction
    def auto_predict(self, data, target_col, model_name):
        """
        Automatically select the right features and predict using the specified model.
        
        Parameters:
            data (pd.DataFrame): The input data
            model_name (str): The name of the model to use
            
        Returns:
            numpy.ndarray: Predictions from the model
        """
        model = self.get_model(model_name)
        
        features = data[['Tg', 'Ta']]
            
        return model.predict(features)
    
    def create_ensemble(self, ensemble_name, model_names, method='average'):
        """
        Create an ensemble of models.
        
        Parameters:
            ensemble_name (str): Name for the new ensemble
            model_names (list): List of model names to include in the ensemble
            method (str): Ensemble method - 'average', 'weighted', or 'stacking'
            
        Returns:
            str: Name of the created ensemble
        """
        # Check that all models exist
        for name in model_names:
            if name not in self.models:
                raise KeyError(f"Model '{name}' not found in the manager")
        
        # Create a simple ensemble class
        class ModelEnsemble:
            def __init__(self, name, models, method='average', weights=None):
                self.name = name
                self.models = models
                self.method = method
                self.weights = weights if weights else [1/len(models)] * len(models)
                self.type = 'ensemble'
            
            def predict(self, data, target_col='Tc'):
                predictions = []
                
                # Get predictions from all models
                for model in self.models:
                    # Select appropriate features if defined
                    features = data[['Tg', 'Ta']]
                    
                    pred = model.predict(features)
                    if pred.ndim > 1 and pred.shape[1] == 1:
                        pred = pred.flatten()
                    
                    predictions.append(pred)
                
                # Combine predictions based on method
                if self.method == 'average':
                    final_pred = np.mean(predictions, axis=0)
                elif self.method == 'weighted':
                    final_pred = np.average(predictions, axis=0, weights=self.weights)
                else:  # Default to average
                    final_pred = np.mean(predictions, axis=0)
                
                return final_pred
        
        # Get the model instances
        models = [self.models[name] for name in model_names]
        
        # Create ensemble
        ensemble = ModelEnsemble(ensemble_name, models, method)
        
        # Add to manager
        self.add_model(ensemble_name, ensemble, group='ensembles')
        
        print(f"Created ensemble '{ensemble_name}' with models: {', '.join(model_names)}")
        
        return ensemble_name

    def save_models_config(self, filename='model_manager_config.json'):
        """
        Save the configuration of all models to a JSON file.
        
        Parameters:
            filename (str): Path to save the configuration file
        """
        config = {
            'models': {},
            'groups': self.model_groups
        }
        
        for name, model in self.models.items():
            config['models'][name] = {
                'type': model.name,
                'model_file': model.model_file if hasattr(model, 'model_file') else '',
                'X_scaler_file': model.X_scaler_file if hasattr(model, 'X_scaler_file') else '',
                'y_scaler_file': model.y_scaler_file if hasattr(model, 'y_scaler_file') else '',
            }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_models_config(self, filename='model_manager_config.json'):
        """
        Load model configurations from a JSON file and create the models.
        
        Parameters:
            filename (str): Path to the configuration file
        """
        with open(filename, 'r') as f:
            config = json.load(f)
        
        # Clear existing models and groups
        self.models = {}
        self.model_groups = {}
        
        # Load groups configuration
        self.model_groups = config['groups']
        
        # Create models
        for name, model_config in config['models'].items():
            self.create_model(
                model_name=name,
                model_type=model_config['type'],
                model_file=model_config['model_file'],
                X_scaler_file=model_config['X_scaler_file'],
                y_scaler_file=model_config['y_scaler_file']
            )