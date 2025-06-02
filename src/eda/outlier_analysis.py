import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Detect and visualize outliers in the dataset
def outlier_analysis(df: pd.DataFrame):
    """
    Perform outlier analysis on the DataFrame.
    This function identifies outliers in the variables 'Tg', 'Ta', and 'Tc' using
    the Interquartile Range (IQR) method and Z-score method.
    It generates boxplots for each variable and prints the number of outliers detected.
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data for outlier analysis.
        It must include the columns 'Tg', 'Ta', and 'Tc'.
    Outputs:
    --------
    1. Prints the number of outliers detected for each variable using the IQR method.
    2. Prints the lower and upper bounds for outlier detection.
    3. Prints the outlier values for each variable.
    4. Generates boxplots for each variable to visualize the distribution and outliers.
    5. Performs Z-score method for outlier detection and prints the number of outliers detected.
    6. Displays the generated plots in the current environment.
    Notes:
    ------
    - The function assumes that the input DataFrame is preprocessed and contains no missing values in the relevant columns.
    - The function uses matplotlib and seaborn for visualization, and scipy.stats for statistical analysis.
    - The generated plots are displayed in the current environment.
    Example:
    --------
    >>> import pandas as pd
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> df = pd.DataFrame({
    ...     'Tg': [36.5, 36.7, 36.8, 36.6, 100],  # 100 is an outlier
    ...     'Ta': [22.1, 22.3, 22.5, 22.4, -50],  # -50 is an outlier
    ...     'Tc': [36.8, 37.0, 37.1, 36.9, 200]   # 200 is an outlier
    ... })
    >>> outlier_analysis(df)
    """

    print("\n--- Outlier Analysis ---")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    variables = ['Tg', 'Ta', 'Tc']
    
    var_outliers = {}
    for i, var in enumerate(variables):
        # Calculate IQR
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
        
        print(f"\n{var} - Outliers detected: {len(outliers)}")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
        if len(outliers) > 0:
            print("Outlier values:")
            print(outliers[var].sort_values().unique())
        
        # Create boxplot
        sns.boxplot(x=df[var], ax=axes[i])
        axes[i].set_title(f'Boxplot of {var}')
        axes[i].set_xlabel(var)

        var_outliers[var] = outliers
    
    plt.tight_layout()
    
    # Z-score method for outlier detection
    print("\nOutlier detection using Z-scores (|z| > 3):")
    
    for var in variables:
        z_scores = np.abs(stats.zscore(df[var]))
        outliers_z = df[z_scores > 3]
        print(f"{var}: {len(outliers_z)} outliers detected by Z-score method")
    
    plt.show()
    return var_outliers