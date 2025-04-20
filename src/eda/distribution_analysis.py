import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.graphics.gofplots import qqplot

from mda import test_normality

def dataframe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of a pandas DataFrame, providing key statistics and insights 
    about each column in the DataFrame.
    The summary includes:
    - Data Type: The data type of each column.
    - Non-Null Count: The number of non-null (non-missing) values in each column.
    - Missing Values: The number of missing (null) values in each column.
    - Distinct Values: The number of unique values in each column.
    - Zeros Count: The number of zero values in each column.
    - Negative Values: The number of negative values in each column.
    - Infinite Values: The number of infinite values in each column.
    Parameters:
        df (pd.DataFrame): The input pandas DataFrame to analyze.
    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each column 
        in the input DataFrame.
    """

    print("\n--- Dataframe Summary ---")

    return pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Values": df.isnull().sum(),
        "Distinct Values": df.nunique(),
        "Zeros Count": (df == 0).sum(),
        "Negative Values": (df < 0).sum(),
        "Infinite Values": np.isinf(df).sum()
    })

def distribution_analysis(df: pd.DataFrame, preprocessing, data_type):
    """
    Perform a comprehensive distribution analysis for specified variables in a DataFrame.
    This function generates histograms, Q-Q plots, and performs statistical tests 
    (Shapiro-Wilk) to assess the normality of the distributions for the variables 
    'Tg', 'Ta', and 'Tc'. The results include visualizations and printed statistical 
    test outcomes. Additionally, the generated plots are saved to a specified directory.
    Args:
        df (pd.DataFrame): The input DataFrame containing the variables 'Tg', 'Ta', and 'Tc'.
    Visualizations:
        - Histograms with KDE (Kernel Density Estimation) for each variable.
        - Q-Q plots to compare the quantiles of the variable against a normal distribution.
    Statistical Tests:
        - Shapiro-Wilk test for normality is performed on a sample of the data (up to 5000 rows).
        - Prints the test statistic and p-value for each variable.
        - Provides an interpretation of whether the distribution is likely normal or not.
    Outputs:
        - Saves the generated plots as a PNG file in the '../reports/figures/' directory.
        - Displays the plots in the current environment.
    Notes:
        - Ensure that the input DataFrame contains the columns 'Tg', 'Ta', and 'Tc'.
        - For large datasets, the Shapiro-Wilk test is performed on a sample to avoid computational overhead.
        - The saved figure is named 'distributions_of_each_variable.png' and has a resolution of 300 DPI.
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Tg': [1.2, 2.3, 3.1, ...],
        ...     'Ta': [0.5, 1.7, 2.8, ...],
        ...     'Tc': [3.4, 2.1, 1.8, ...]
        ... })
        >>> distribution_analysis(df)
    """

    print("\n--- Distribution Analysis ---")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    variables = ['Tg', 'Ta', 'Tc']
    
    for i, var in enumerate(variables):
        # Histogram
        sns.histplot(df[var], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribution of {var}')
        axes[i, 0].set_xlabel(var)

        # QQ plot
        qqplot(df[var], line='s', ax=axes[i, 1])
        axes[i, 1].set_title(f'Q-Q Plot of {var}')
        
    plt.tight_layout()

    # Statistical tests for normality
    print("\nNormality Tests (Shapiro-Wilk):")
    for var in variables:
        # For large datasets, take a sample
        sample = df[var].sample(min(5000, len(df)))
        res = test_normality(sample)
        print(res)
        if not res['conclusion']:
            print(f"  The distribution of {var} is likely not normal (p < 0.05)")
        else:
            print(f"  The distribution of {var} appears to be normal (p >= 0.05)")
    
    # Save the figure automatically to the specified directory
    figure_path = f'../reports/figures/eda/{preprocessing}/{data_type}'
    os.makedirs(figure_path, exist_ok=True)
    plt.savefig(f'{figure_path}/distributions_of_each_variable.png', dpi=300, bbox_inches='tight')

    plt.show()

def basic_statistics(train_df, test_df):
    """
    Calculate and compare basic statistics between training and test sets.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    """
    print("\n=== Basic Statistics Comparison ===")
    
    # Assuming your features are 'glabella_temp' and 'ambient_temp'
    features = ['Tg', 'Ta', 'Tc']
    
    for feature in features:
        if feature in train_df.columns and feature in test_df.columns:
            print(f"\nFeature: {feature}")
            
            train_stats = train_df[feature].describe()
            test_stats = test_df[feature].describe()
            
            # Create comparison DataFrame
            stats_comparison = pd.DataFrame({
                'Training': train_stats,
                'Test': test_stats,
                'Difference': train_stats - test_stats,
                'Percent Diff': ((train_stats - test_stats) / train_stats) * 100
            })
            
            print(stats_comparison)

def distribution_plots(train_df, test_df):
    """
    Generate distribution plots comparing training and test data.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    """
    features = ['Tg', 'Ta', 'Tc']
    
    fig, axes = plt.subplots(len(features), 2, figsize=(15, 5 * len(features)))
    
    for i, feature in enumerate(features):
        if feature in train_df.columns and feature in test_df.columns:
            # Histogram
            sns.histplot(train_df[feature], color='blue', alpha=0.5, label='Training', ax=axes[i, 0])
            sns.histplot(test_df[feature], color='red', alpha=0.5, label='Test', ax=axes[i, 0])
            axes[i, 0].set_title(f'Histogram of {feature}')
            axes[i, 0].legend()
            
            # KDE plot
            sns.kdeplot(train_df[feature], color='blue', label='Training', ax=axes[i, 1])
            sns.kdeplot(test_df[feature], color='red', label='Test', ax=axes[i, 1])
            axes[i, 1].set_title(f'Density of {feature}')
            axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.close()
    
    # Add bivariate plots for feature relationships
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot of glabella vs. core temp, if available
    if all(col in train_df.columns for col in ['glabella_temp', 'core_temp']) and \
       all(col in test_df.columns for col in ['glabella_temp', 'core_temp']):
        plt.scatter(train_df['glabella_temp'], train_df['core_temp'], alpha=0.5, 
                    color='blue', label='Training')
        plt.scatter(test_df['glabella_temp'], test_df['core_temp'], alpha=0.5, 
                    color='red', label='Test')
        plt.xlabel('Glabella Temperature')
        plt.ylabel('Core Temperature')
        plt.title('Relationship between Glabella and Core Temperature')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('temperature_relationship.png')
        plt.close()

def kolmogorov_smirnov_test(train_df, test_df):
    """
    Perform the two-sample Kolmogorov-Smirnov test to compare distributions.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    
    Returns:
    - results_df: DataFrame containing KS test results
    """
    print("\n=== Kolmogorov-Smirnov Test Results ===")
    features = ['Tg', 'Ta', 'Tc']
    results = []
    
    for feature in features:
        if feature in train_df.columns and feature in test_df.columns:
            ks_statistic, p_value = stats.ks_2samp(train_df[feature], test_df[feature])
            
            interpretation = "Same distribution" if p_value > 0.05 else "Different distribution"
            
            results.append({
                'Feature': feature,
                'KS Statistic': ks_statistic,
                'p-value': p_value,
                'Interpretation (α=0.05)': interpretation
            })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

def permutation_test(train_df, test_df, n_permutations=1000):
    """
    Perform a permutation test to compare distributions.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    - n_permutations: Number of permutations to perform
    """
    print("\n=== Permutation Test Results ===")
    features = ['Tg', 'Ta', 'Tc']
    
    for feature in features:
        if feature in train_df.columns and feature in test_df.columns:
            # Calculate original mean difference
            original_diff = abs(train_df[feature].mean() - test_df[feature].mean())
            
            # Combine data
            combined = np.concatenate([train_df[feature].values, test_df[feature].values])
            n_train = len(train_df[feature])
            
            # Permutation test
            count = 0
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_train = combined[:n_train]
                perm_test = combined[n_train:]
                perm_diff = abs(perm_train.mean() - perm_test.mean())
                
                if perm_diff >= original_diff:
                    count += 1
            
            p_value = count / n_permutations
            interpretation = "Same distribution" if p_value > 0.05 else "Different distribution"
            
            print(f"\nFeature: {feature}")
            print(f"Original mean difference: {original_diff:.4f}")
            print(f"Permutation test p-value: {p_value:.4f}")
            print(f"Interpretation (α=0.05): {interpretation}")

def dimensionality_reduction_visualization(train_df, test_df):
    """
    Visualize data in lower dimensions using PCA and t-SNE.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    """
    features = ['Tg', 'Tc']
    
    if all(feature in train_df.columns for feature in features) and \
       all(feature in test_df.columns for feature in features):
        
        # Extract features
        X_train = train_df[features].values
        X_test = test_df[features].values
        
        # Combine data for transformation
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.array(['Train'] * len(X_train) + ['Test'] * len(X_test))
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_combined)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA plot
        for label, color in zip(['Train', 'Test'], ['blue', 'red']):
            mask = y_combined == label
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.5)
        
        axes[0].set_title('PCA Visualization')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # t-SNE plot
        for label, color in zip(['Train', 'Test'], ['blue', 'red']):
            mask = y_combined == label
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=label, alpha=0.5)
        
        axes[1].set_title('t-SNE Visualization')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction.png')
        plt.close()

def calculate_mmd(X, Y, gamma=1.0):
    """
    Calculate Maximum Mean Discrepancy (MMD) between two distributions.
    
    Parameters:
    - X: First sample
    - Y: Second sample
    - gamma: RBF kernel parameter
    
    Returns:
    - MMD value
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    
    rx = np.diag(XX).reshape(-1, 1)
    ry = np.diag(YY).reshape(-1, 1)
    
    dxx = rx + rx.T - 2 * XX
    dyy = ry + ry.T - 2 * YY
    dxy = rx + ry.T - 2 * XY
    
    XX_rbf = np.exp(-gamma * dxx)
    YY_rbf = np.exp(-gamma * dyy)
    XY_rbf = np.exp(-gamma * dxy)
    
    m = X.shape[0]
    n = Y.shape[0]
    
    mmd = np.sqrt(np.sum(XX_rbf) / (m * m) + np.sum(YY_rbf) / (n * n) - 2 * np.sum(XY_rbf) / (m * n))
    
    return mmd

def mmd_test(train_df, test_df):
    """
    Perform Maximum Mean Discrepancy test.
    
    Parameters:
    - train_df: DataFrame containing training data
    - test_df: DataFrame containing test data
    """
    print("\n=== Maximum Mean Discrepancy Test Results ===")
    features = ['Tg', 'Ta']
    
    if all(feature in train_df.columns for feature in features) and \
       all(feature in test_df.columns for feature in features):
        
        # Scale features
        X_train = train_df[features].values
        X_test = test_df[features].values
        
        # Calculate MMD
        mmd_value = calculate_mmd(X_train, X_test)
        
        # MMD permutation test
        n_permutations = 100
        combined = np.vstack([X_train, X_test])
        n_train = X_train.shape[0]
        
        mmd_permutations = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_train = combined[:n_train]
            perm_test = combined[n_train:]
            mmd_permutations.append(calculate_mmd(perm_train, perm_test))
        
        # Calculate p-value
        p_value = np.mean(np.array(mmd_permutations) >= mmd_value)
        
        print(f"MMD value: {mmd_value:.6f}")
        print(f"Permutation test p-value: {p_value:.4f}")
        print(f"Interpretation (α=0.05): {'Different distribution' if p_value <= 0.05 else 'Same distribution'}")