import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Analyze the correlations between variables
def correlation_analysis(df: pd.DataFrame, data_type):
    """
    Perform a comprehensive correlation analysis on a given DataFrame.
    This function calculates and visualizes the correlation between variables in the provided DataFrame.
    It generates a correlation matrix, scatter plot matrix, and specific scatter plots with regression lines.
    Additionally, it computes and displays statistical correlation coefficients (Pearson and Spearman) 
    along with their significance levels.
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data for correlation analysis. 
        It must include the columns 'Tg', 'Ta', and 'Tc' for specific scatter plots and regression analysis.
    Outputs:
    --------
    1. Prints the correlation matrix rounded to 3 decimal places.
    2. Saves the following visualizations to the '../reports/figures/' directory:
        - Correlation matrix heatmap as 'correlation_matrix.png'.
        - Scatter plot matrix as 'scatter_plot_matrix.png'.
        - Scatter plots with regression lines as 'scatter_plot_with_regression_lines.png'.
    3. Prints Pearson and Spearman correlation coefficients for 'Tg' and 'Ta' against 'Tc', 
        along with their p-values and significance interpretation.
    Visualizations:
    ---------------
    - Correlation Matrix: A heatmap showing the pairwise correlation coefficients between variables.
    - Scatter Plot Matrix: A grid of scatter plots for pairwise relationships, with kernel density estimates on the diagonal.
    - Scatter Plots with Regression Lines:
        - 'Tg' (Glabella Temperature) vs 'Tc' (Core Body Temperature), with horizontal reference lines at y=36.6 and y=37.2.
        - 'Ta' (Ambient Temperature) vs 'Tc' (Core Body Temperature).
    Statistical Analysis:
    ---------------------
    - Pearson Correlation Coefficients: Measures the linear relationship between variables.
    - Spearman Rank Correlation Coefficients: Measures the monotonic relationship between variables.
    Notes:
    ------
    - The function assumes that the input DataFrame is preprocessed and contains no missing values in the relevant columns.
    - The saved figures are stored in the relative path '../reports/figures/'. Ensure the directory exists before running the function.
    - The function uses matplotlib and seaborn for visualization, and scipy.stats for statistical analysis.
    Example:
    --------
    >>> import pandas as pd
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> df = pd.DataFrame({
    ...     'Tg': [36.5, 36.7, 36.8, 36.6],
    ...     'Ta': [22.1, 22.3, 22.5, 22.4],
    ...     'Tc': [36.8, 37.0, 37.1, 36.9]
    ... })
    >>> correlation_analysis(df)
    """

    print("\n--- Correlation Analysis ---")

    # Calculate correlation matrix
    corr_matrix = df.corr().round(3)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')

    figure_path = f'../reports/figures/non_processed/{data_type}'
    os.makedirs(figure_path, exist_ok=True)

    # Save the figure automatically to the specified directory
    plt.savefig(f'{figure_path}/correlation_matrix_cleaned.png', dpi=300, bbox_inches='tight')

    # Scatter plot matrix
    plt.figure(figsize=(15, 10))
    scatter_matrix = sns.pairplot(df, kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5})
    scatter_matrix.fig.suptitle('Scatter Plot Matrix', y=1.02, fontsize=16)

    # Save the figure automatically to the specified directory

    plt.savefig(f'{figure_path}/scatter_plot_matrix_cleaned.png', dpi=300, bbox_inches='tight')

    # Specific scatter plots with regression lines
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tg vs Tc
    sns.regplot(x='Tg', y='Tc', data=df, ax=axes[0], scatter_kws={'alpha': 0.5})
    axes[0].set_title('Glabella Temperature vs Core Body Temperature')

    # Add horizontal lines at y=36.6 and y=37.2
    axes[0].axhline(y=36.6, color='red', linestyle='--', linewidth=1, label='y=36.6')
    axes[0].axhline(y=37.2, color='blue', linestyle='--', linewidth=1, label='y=37.2')
    axes[0].legend()

    # Ta vs Tc
    sns.regplot(x='Ta', y='Tc', data=df, ax=axes[1], scatter_kws={'alpha': 0.5})
    axes[1].set_title('Ambient Temperature vs Core Body Temperature')

    plt.tight_layout()

    # Save the figure automatically to the specified directory
    plt.savefig(f'{figure_path}/scatter_plot_with_regression_lines_cleaned.png', dpi=300, bbox_inches='tight')  

    # Calculate and print statistical correlations
    print("\nPearson Correlation Coefficients:")
    for var1 in ['Tg', 'Ta']:
        r, p = stats.pearsonr(df[var1], df['Tc'])
        print(f"{var1} vs Tc: r = {r:.4f}, p-value = {p:.4e}")
        if p < 0.05:
            print(f"  The correlation between {var1} and Tc is statistically significant")
        else:
            print(f"  The correlation between {var1} and Tc is not statistically significant")

    print("\nSpearman Rank Correlation Coefficients:")
    for var1 in ['Tg', 'Ta']:
        rho, p = stats.spearmanr(df[var1], df['Tc'])
        print(f"{var1} vs Tc: rho = {rho:.4f}, p-value = {p:.4e}")

    plt.show()