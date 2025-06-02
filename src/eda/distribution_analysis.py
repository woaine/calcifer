import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
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