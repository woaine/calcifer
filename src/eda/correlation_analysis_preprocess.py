import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Analyze the correlations between variables in the preprocessed dataset
# This function is similar to the above but assumes that the dataset has been preprocessed (outliers were removed).
def correlation_analysis_preprocess(df: pd.DataFrame, data_type):
    """
    Perform the same correlation analysis on the preprocessed dataset.
    """

    print("\n--- Correlation Analysis on Preprocessed Dataset ---")

    # Calculate correlation matrix
    corr_matrix = df.corr().round(3)
    print("\nCorrelation Matrix on Preprocessed Dataset:")
    print(corr_matrix)

    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix on Preprocessed Dataset')

    figure_path = f'../reports/figures/preprocessed/{data_type}'
    os.makedirs(figure_path, exist_ok=True)

    # Save the figure automatically to the specified directory
    plt.savefig(f'{figure_path}/correlation_matrix_preprocessed.png', dpi=300, bbox_inches='tight')

    # Scatter plot matrix
    plt.figure(figsize=(15, 10))
    scatter_matrix = sns.pairplot(df, kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5})
    scatter_matrix.fig.suptitle('Scatter Plot Matrix on Preprocessed Dataset', y=1.02, fontsize=16)

    # Save the figure automatically to the specified directory
    plt.savefig(f'{figure_path}/scatter_plot_matrix_preprocessed.png', dpi=300, bbox_inches='tight')

    # Specific scatter plots with regression lines
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tg vs Tc
    sns.regplot(x='Tg', y='Tc', data=df, ax=axes[0], scatter_kws={'alpha': 0.5})
    axes[0].set_title('Glabella Temperature vs Core Body Temperature (Preprocessed)')

    # Add horizontal lines at y=36.6 and y=37.2
    axes[0].axhline(y=36.6, color='red', linestyle='--', linewidth=1, label='y=36.6')
    axes[0].axhline(y=37.2, color='blue', linestyle='--', linewidth=1, label='y=37.2')
    axes[0].legend()

    # Ta vs Tc
    sns.regplot(x='Ta', y='Tc', data=df, ax=axes[1], scatter_kws={'alpha': 0.5})
    axes[1].set_title('Ambient Temperature vs Core Body Temperature (Preprocessed)')

    plt.tight_layout()

    # Save the figure automatically to the specified directory
    plt.savefig(f'{figure_path}/scatter_plot_with_regression_lines_preprocessed.png', dpi=300, bbox_inches='tight')  

    # Calculate and print statistical correlations
    print("\nPearson Correlation Coefficients (Preprocessed):")
    for var1 in ['Tg', 'Ta']:
        r, p = stats.pearsonr(df[var1], df['Tc'])
        print(f"{var1} vs Tc: r = {r:.4f}, p-value = {p:.4e}")
        if p < 0.05:
            print(f"  The correlation between {var1} and Tc is statistically significant")
        else:
            print(f"  The correlation between {var1} and Tc is not statistically significant")

    print("\nSpearman Rank Correlation Coefficients (Preprocessed):")
    for var1 in ['Tg', 'Ta']:
        rho, p = stats.spearmanr(df[var1], df['Tc'])
        print(f"{var1} vs Tc: rho = {rho:.4f}, p-value = {p:.4e}")

    plt.show()