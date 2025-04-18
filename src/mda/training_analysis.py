import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from itertools import cycle

def training_progress(df: pd.DataFrame, metric, metric_name, scale, preprocessing, data_type, best_models_df=None, zoom_range=(0, 1), legend_columns=2):
    """
    Plot training progress of models across epochs for different configurations.
    Models are color-grouped by the number of layers, with different hues for each activation and batch size.
    Highlights the peak epoch of each model configuration based on a reference dataframe.
    Creates a dual-subplot visualization with a zoomed view of the 0-1 range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model training data with columns:
        'epoch', 'layers', 'activation', 'batch_size', and a metric column (e.g., 'val_loss', 'accuracy')
    metric : str
        Name of the metric column in the DataFrame to plot (e.g., 'val_loss', 'accuracy')
    metric_name : str
        Display name for the metric on the plot
    best_models_df : pandas.DataFrame, optional
        DataFrame containing the best epoch for each model configuration with columns:
        'layers', 'activation', 'batch_size', 'best_epoch', and the metric column
    zoom_range : tuple, optional
        Range for the zoomed view (default is (0, 1))
    legend_columns : int, optional
        Number of columns in the legend (default is 2)
    """
    
    # Close any existing figures to prevent interference
    plt.close()
    
    # Create a figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 3]})
    
    # Extract unique values for grouping
    unique_layers = sorted(df['layers'].unique())
    unique_activations = sorted(df['activation'].unique())
    unique_batch_sizes = sorted(df['batch_size'].unique())
    
    # Generate distinct base colors for each layer group
    layer_colors = {}
    # Using distinct base colors for different layer groups
    base_colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'olive']
    
    for i, layer in enumerate(unique_layers):
        # Assign a base color to each layer group (cycling if we have more layers than colors)
        layer_colors[layer] = base_colors[i % len(base_colors)]
    
    # Get unique model configurations
    model_configs = df.groupby(['layers', 'activation', 'batch_size'])
    
    # Keep track of configuration names and colors for the legend
    config_names = []
    config_colors = []
    
    # Find if we're looking for min or max values
    find_min = 'loss' in metric.lower()
    
    # Variables to keep track of the global best model
    global_best_value = float('inf') if find_min else float('-inf')
    global_best = None
    
    # If we have a reference dataframe, determine the global best from it
    if best_models_df is not None:
        # Find the row with the best metric value in the reference dataframe
        if find_min:
            best_idx = best_models_df[metric].idxmin() if metric in best_models_df.columns else None
        else:
            best_idx = best_models_df[metric].idxmax() if metric in best_models_df.columns else None
            
        # If we found a valid index, extract the global best configuration
        if best_idx is not None:
            best_row = best_models_df.loc[best_idx]
            global_best_config = (best_row['layers'], best_row['activation'], best_row['batch_size'])
    
    # List to collect all best models data for returning
    best_models_data = []
    
    # Create a dictionary to store variation indices for each layer
    layer_variations = {layer: 0 for layer in unique_layers}
    
    # Dictionary to store model lines for each configuration
    model_lines = {}
    
    # Iterate through each model configuration
    for (layers, activation, batch_size), group_df in model_configs:
        # Sort by epoch
        group_df = group_df.sort_values('epoch')
        
        # Get the epochs and metric values
        epochs = group_df['epoch'].values
        values = group_df[metric].values
        
        # Get base color for this layer
        base_color = layer_colors[layers]
        
        # Create a variation of this color based on activation and batch size
        # Get current variation index for this layer and increment it
        variation_idx = layer_variations[layers]
        layer_variations[layers] += 1
        
        # Create color variations within the same hue family
        base_rgb = list(mcolors.to_rgb(base_color))
        
        # Adjust saturation and brightness based on the variation index
        # This creates variations within the same color family
        h, l, s = tuple(mcolors.rgb_to_hsv(base_rgb))
        
        # Vary saturation and lightness to create distinct but related colors
        s_variation = 0.7 + (variation_idx % 4) * 0.1  # Vary saturation
        l_variation = 0.3 + (variation_idx // 4) * 0.15  # Vary lightness
        
        # Ensure lightness doesn't go too high (too light) or too low (too dark)
        l_variation = max(0.25, min(0.75, l_variation))
        
        # Convert back to RGB
        varied_color = mcolors.hsv_to_rgb([h, l_variation, s_variation])
        
        # Plot the line for this model configuration on both subplots
        config_name = f"Layers: {layers}, Activation: {activation}, Batch: {batch_size}"
        line1, = ax1.plot(epochs, values, label=config_name, color=varied_color, linewidth=2, alpha=0.8)
        line2, = ax2.plot(epochs, values, color=varied_color, linewidth=2, alpha=0.8)
        
        # Store the information for legend grouping
        config_names.append(config_name)
        config_colors.append(varied_color)
        model_lines[config_name] = line1
        
        # Determine the best epoch for this configuration
        best_epoch = None
        best_value = None
        
        # If best_models_df is provided, find the best epoch from it
        if best_models_df is not None:
            # Filter the best_models_df to find the entry for this configuration
            best_model_row = best_models_df[(best_models_df['layers'] == layers) & 
                                           (best_models_df['activation'] == activation) & 
                                           (best_models_df['batch_size'] == batch_size)]
            
            if not best_model_row.empty:
                best_epoch = best_model_row['epoch'].values[0]
                # Get the metric value at this epoch from the original dataframe
                epoch_value = group_df.loc[group_df['epoch'] == best_epoch, metric]
                if not epoch_value.empty:
                    best_value = epoch_value.values[0]
                else:
                    # If the epoch from best_models_df is not in the original df, find the closest
                    closest_epoch = group_df['epoch'].iloc[(group_df['epoch'] - best_epoch).abs().argsort()[0]]
                    best_value = group_df.loc[group_df['epoch'] == closest_epoch, metric].values[0]
                    best_epoch = closest_epoch
        
        # If not found in best_models_df, compute the best epoch based on the metric
        if best_epoch is None:
            if find_min:
                best_idx = group_df[metric].idxmin()
                best_epoch = group_df.loc[best_idx, 'epoch']
                best_value = group_df[metric].min()
            else:
                best_idx = group_df[metric].idxmax()
                best_epoch = group_df.loc[best_idx, 'epoch']
                best_value = group_df[metric].max()
        
        # Highlight the best epoch for this configuration on both subplots
        ax1.scatter(best_epoch, best_value, s=60, color=varied_color, 
                   edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
        ax2.scatter(best_epoch, best_value, s=60, color=varied_color, 
                   edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
        
        # Store the best model data
        current_best = (layers, activation, batch_size, best_epoch, best_value, varied_color)
        best_models_data.append(current_best)
        
        # Check if this is the global best model
        # If we have a reference dataframe with a global best already identified
        if best_models_df is not None and 'global_best_config' in locals():
            if (layers, activation, batch_size) == global_best_config:
                global_best = current_best
                global_best_value = best_value
        # Otherwise find it based on the metric values
        elif (find_min and best_value < global_best_value) or (not find_min and best_value > global_best_value):
            global_best_value = best_value
            global_best = current_best
    
    # Emphasize the global best model with a larger marker and annotation
    if global_best:
        layers, activation, batch_size, best_epoch, best_value, color = global_best
        
        # Add a larger marker for the global best
        ax1.scatter(best_epoch, best_value, s=100, color=color, 
                   edgecolor='white', linewidth=2, zorder=6)
        ax2.scatter(best_epoch, best_value, s=100, color=color, 
                   edgecolor='white', linewidth=2, zorder=6)
        
        # # Add annotation for the global best model
        # optimal_config_str = f"Layers: {layers}, Act: {activation}, Batch: {batch_size}"
        # annotation_text = f"Global Best: {optimal_config_str}\nEpoch {int(best_epoch)}: {best_value:.4f}"
        
        # if zoom_range[0] <= best_value <= zoom_range[1]:
        #     # If optimal value is in zoomed range, add annotation to the zoomed subplot
        #     ax2.annotate(annotation_text,
        #                 xy=(best_epoch, best_value),
        #                 xytext=(best_epoch + 10, best_value + 0.05),
        #                 fontsize=10,
        #                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        # else:
        #     # Otherwise add to the upper subplot
        #     ax1.annotate(annotation_text,
        #                 xy=(best_epoch, best_value),
        #                 xytext=(best_epoch + 10, best_value * 1.05),
        #                 fontsize=10,
        #                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Determine overall y-axis limits from the data
    y_min = df[metric].min()
    y_max = df[metric].max()
    
    # Set the y-limits for each subplot to create the "zoomed" effect
    # First subplot: Show the full range of the data
    if y_min < 0:
        ax1.set_ylim(y_min, y_max)
    else:
        ax1.set_ylim(0, y_max)
    
    # Second subplot: Show only the specified zoom range
    ax2.set_ylim(zoom_range[0], zoom_range[1])
    
    # X-axis settings (only need to set on the bottom subplot due to sharex=True)
    ax2.set_xlim(1, 300)
    ax2.set_xticks(np.arange(0, 301, 25))
    
    # Grid settings
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Only show x-label on the bottom subplot
    ax2.set_xlabel('Epoch', fontsize=12)
    
    # Y-labels
    ax1.set_ylabel(metric_name, fontsize=12)
    ax2.set_ylabel(f"{metric_name} (Zoomed {zoom_range[0]}-{zoom_range[1]})", fontsize=12)
    
    # Title only on top subplot
    ax1.set_title(f'Validation {metric_name} vs Epoch', fontsize=14)
    
    # Create a grouped legend that organizes by number of layers
    # Create layer group patches for legend
    legend_elements = []
    
    # Add group headers for each layer
    for layer in unique_layers:
        # Add a header for this layer group
        base_color = layer_colors[layer]
        legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                                     label=f"---- Layer {layer} Models ----"))
        
        # Add each model configuration that has this layer
        for i, config_name in enumerate(config_names):
            if f"Layers: {layer}," in config_name:
                legend_elements.append(model_lines[config_name])
    
    # Add legend with grouping
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10, framealpha=0.9, ncol=legend_columns)
    
    # Add a broken axis effect
    d = .015  # size of the diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # bottom-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-right diagonal
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
    
    # Save the figure automatically to the specified directory
    plt.savefig(f'../reports/figures/{metric_name}_{scale}_{preprocessing}_{data_type}_training_progress.png', dpi=300, bbox_inches='tight') 

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, right=0.8)  # Reduce space between subplots and make room for legend
    plt.show()

def training_loss_r2(df, scale, type='validation', loss_zoom_range=None, r2_zoom_range=None):
    """
    Plot either training or validation loss alongside R2 scores with zoomed y-axis versions
    for multiple model configurations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns 'loss', 'val_loss', 'r2_score', 'val_r2_score',
        and model configuration columns 'scale', 'preprocessing', 'data_type'
    scale : str
        Scale type to filter data ('standardized', 'minmax', etc.)
    type : str, optional
        Type of data to plot: 'training' or 'validation' (default: 'validation')
    loss_zoom_range : tuple, optional
        Tuple of (min_y, max_y) for zooming in on the loss plot y-axis
        If None, auto-calculates a reasonable zoom range based on the data
    r2_zoom_range : tuple, optional
        Tuple of (min_y, max_y) for zooming in on the R2 score plot y-axis
        If None, auto-calculates a reasonable zoom range based on the data
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Filter by scale
    filtered_df = df[df['scale'] == scale]
    
    # Validate data_type parameter
    if type.lower() not in ['training', 'validation']:
        raise ValueError("type must be either 'training' or 'validation'")
    
    # Set column names based on type
    if type.lower() == 'training':
        loss_col = 'loss'
        r2_col = 'r2_score'
        title_prefix = 'Training'
    else:  # validation
        loss_col = 'val_loss'
        r2_col = 'val_r2_score'
        title_prefix = 'Validation'
    
    # Define colors for different model configurations
    colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']
    
    # Create a figure with four subplots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, 
                           gridspec_kw={'height_ratios': [1, 3]})
    
    # Get unique model configurations
    model_configs = filtered_df[['preprocessing', 'data_type']].drop_duplicates()
    
    # Calculate overall minimum and maximum for loss and r2 across all models
    all_loss_values = filtered_df[loss_col].values
    all_r2_values = filtered_df[r2_col].values
    
    # Calculate default zoom ranges if not provided
    if loss_zoom_range is None:
        # Calculate a reasonable y-axis zoom range for loss
        min_loss = np.min(all_loss_values)
        median_loss = np.median(all_loss_values)
        
        # Create a range with some padding
        loss_range = median_loss - min_loss
        loss_zoom_range = (
            max(0, min_loss - 0.1 * loss_range),  # Lower bound (ensure positive)
            min_loss + loss_range * 1.5  # Upper bound with padding
        )
    
    if r2_zoom_range is None:
        # Calculate a reasonable y-axis zoom range for R2
        min_r2 = np.min(all_r2_values)
        max_r2 = np.max(all_r2_values)
        r2_range = max_r2 - min_r2
        
        # Ensure we include zero as a reference if it's not too far
        if min_r2 > 0 and min_r2 < 0.3:
            min_r2_zoom = 0
        else:
            min_r2_zoom = min_r2 - 0.1 * r2_range
            
        r2_zoom_range = (
            min_r2_zoom,
            max_r2 + 0.1 * r2_range
        )
    
    # Plot each model configuration with different colors
    for i, (_, config) in enumerate(model_configs.iterrows()):
        # Get color for this model (cycling through colors if needed)
        color_idx = i % len(colors)
        current_color = colors[color_idx]
        
        # Filter data for this model configuration
        model_mask = ((filtered_df['preprocessing'] == config['preprocessing']) & 
                     (filtered_df['data_type'] == config['data_type']))
        model_data = filtered_df[model_mask].sort_values('epoch')
        
        # Generate epoch numbers for this model
        epochs = np.arange(1, len(model_data) + 1)
        
        # Create label for the legend
        model_label = f"{'Preprocessed ' if config['preprocessing'] == 'preprocessed' else ''}{config['data_type'].capitalize()}"
        
        # Plot in all 4 subplots
        # Top left - Loss
        axs[0, 0].plot(epochs, model_data[loss_col], color=current_color, label=model_label)
        
        # Top right - R2
        axs[0, 1].plot(epochs, model_data[r2_col], color=current_color, label=model_label)
        
        # Bottom left - Zoomed Loss
        axs[1, 0].plot(epochs, model_data[loss_col], color=current_color, label=model_label)
        
        # Bottom right - Zoomed R2
        axs[1, 1].plot(epochs, model_data[r2_col], color=current_color, label=model_label)
    
    # Set titles and formatting for all subplots
    axs[0, 0].set_title(f'{title_prefix} Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axs[0, 1].set_title(f'{title_prefix} R2 Score')
    axs[0, 1].set_ylabel('R2 Score')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set the zoomed y-range for loss and titles
    axs[1, 0].set_ylim(loss_zoom_range)
    axs[1, 0].set_title(f'Zoomed Y-Axis: Loss Range [{loss_zoom_range[0]:.4f}, {loss_zoom_range[1]:.4f}]')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Set the zoomed y-range for R2 and titles
    axs[1, 1].set_ylim(r2_zoom_range)
    axs[1, 1].set_title(f'Zoomed Y-Axis: R2 Range [{r2_zoom_range[0]:.4f}, {r2_zoom_range[1]:.4f}]')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('R2 Score')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add a legend to each subplot
    for ax in axs.flat:
        ax.legend(loc='best')
    
    # Save the figure automatically to the specified directory
    plt.savefig(f'../reports/figures/{type}_{scale}_model_comparison.png', dpi=300, bbox_inches='tight') 

    # Add an overall title
    scaler_type = 'Standard Scaler' if scale == 'standardized' else 'MinMax Scaler'
    plt.suptitle(f'{title_prefix} Metrics with {scaler_type}', fontsize=16, y=0.98)
    
    # Adjust the layout with more space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
    plt.show()

def model_metrics(df, scale, figsize=(20, 10)):
    """
    Creates a 2x4 subplot figure showing training and validation metrics (loss and r2_score)
    for 4 different models.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing model configurations and their metrics across epochs.
        Expected columns: 'scale', 'preprocessing', 'data_type', 'epoch',
                         'loss', 'val_loss', 'r2_score', 'val_r2_score',
                         'layers', 'activation', 'batch_size'
    
    scale : str
        The scale type to filter the DataFrame by (e.g., 'standardized', 'minmax')
        
    figsize : tuple, default=(20, 10)
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig, axes : tuple
        The matplotlib figure and axes objects
    """
    # Get unique model configurations for the specified scale
    df = df[df['scale'] == scale]
    model_configs = df[['scale', 'preprocessing', 'data_type']].drop_duplicates()

    # Ensure we have exactly 4 models
    if len(model_configs) != 4:
        raise ValueError(f"Expected 4 unique model configurations, but found {len(model_configs)}")
    
    # Create figure with 2x4 subplots (2 rows for loss and r2_score, 4 columns for models)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # Define metrics to plot
    metrics = ['loss', 'r2_score']
    metric_colors = {
        'loss': {'train': '#4477AA', 'val': '#EE6677'},
        'r2_score': {'train': '#228833', 'val': '#CCBB44'}
    }
    
    # For each model configuration, plot its metrics
    for i, (_, config) in enumerate(model_configs.iterrows()):
        # Filter data for this model
        mask = ((df['scale'] == config['scale']) & 
                (df['preprocessing'] == config['preprocessing']) & 
                (df['data_type'] == config['data_type']))
        model_data = df[mask].sort_values('epoch')
        
        # Specification for title
        specification = f"With {'Preprocessed ' if config['preprocessing'] == 'preprocessed' else ''}{config['data_type'].capitalize()} Data"
        model_name = f"{specification}\nLayers: {model_data['layers'].iloc[0]}\nAct: {model_data['activation'].iloc[0]}\nBatch: {model_data['batch_size'].iloc[0]}"
        
        # Plot both metrics for this model
        for j, metric in enumerate(metrics):
            ax = axes[j, i]
            
            # Plot training and validation metrics
            train_line, = ax.plot(model_data['epoch'], model_data[metric], 
                                 color=metric_colors[metric]['train'], 
                                 label=f'Training {metric}')
            val_line, = ax.plot(model_data['epoch'], model_data[f'val_{metric}'], 
                               color=metric_colors[metric]['val'], 
                               label=f'Validation {metric}')
            
            # Set labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Only set model title for the top row
            if j == 0:
                ax.set_title(model_name)
            
            # Optimize y-axis zoom level
            y_values = np.concatenate([model_data[metric].values, model_data[f'val_{metric}'].values])
            y_min, y_max = np.min(y_values), np.max(y_values)
            y_range = max(y_max - y_min, 1e-5)  # Prevent division by zero
            padding = 0.1 * y_range  # 10% padding
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legends for both metrics
    loss_handles = [plt.Line2D([0], [0], color=metric_colors['loss']['train'], label='Training Loss'),
                   plt.Line2D([0], [0], color=metric_colors['loss']['val'], label='Validation Loss')]
    
    r2_handles = [plt.Line2D([0], [0], color=metric_colors['r2_score']['train'], label='Training R² Score'),
                 plt.Line2D([0], [0], color=metric_colors['r2_score']['val'], label='Validation R² Score')]
    
    # Add the legends
    fig.legend(handles=loss_handles + r2_handles, 
              loc='upper center', 
              ncol=4, 
              bbox_to_anchor=(0.5, 0.02))
    
    # Set the title for the whole figure
    scaler_type = 'Standard Scaler' if scale == 'standardized' else 'MinMax Scaler'
    fig.suptitle(f"Model Performance with {scaler_type}", fontsize=16, y=0.98)
    
    # Save the figures automatically
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'../reports/figures/{scale}_training_validation_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def multi_model_metrics(data, x='metric', group_by=None, compare_by=None,
                            figsize=(12, 8), title=None, 
                            sort_metrics=True, bar_width=0.8, colors=None):
    """
    Plot metrics with confidence intervals for multiple model configurations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing columns for:
        - 'metric': The name of the metric
        - 'value': The value of the metric
        - 'lower_bound': Lower bound of confidence interval
        - 'upper_bound': Upper bound of confidence interval
        - Additional columns that define model configurations 
          (e.g., 'scale', 'preprocessing', 'data_type')
    
    x : str, optional
        Column to use for x-axis. Default is 'metric'.
    
    group_by : str or list, optional
        Column(s) to group the data by (appears as separate groups on x-axis).
        Default is None (uses 'metric').
    
    compare_by : str or list, optional
        Column(s) to compare (appears as different colored bars within each group).
        Default is None (plots all data together).
    
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (12, 8).
    
    title : str, optional
        Plot title. If None, a title will be generated based on the data.
    
    sort_metrics : bool, optional
        Whether to sort metrics alphabetically. Default is True.
    
    bar_width : float, optional
        Width of bars relative to group width. Default is 0.8.
    
    colors : list, optional
        List of colors for different configurations. If None, default colors are used.
    
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects
    """
    # Convert to DataFrame if it's a dict
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Validate required columns
    required_cols = ['metric', 'value', 'lower_bound', 'upper_bound']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data is missing required columns: {missing_cols}")
    
    # Set default group_by if not provided
    if group_by is None:
        group_by = x
    
    # Convert single column names to lists for consistent handling
    if isinstance(group_by, str):
        group_by = [group_by]
    
    # If compare_by is not specified, we'll plot everything together
    if compare_by is None:
        # Find configuration columns (not in required_cols and not in group_by)
        config_cols = [col for col in df.columns 
                      if col not in required_cols and col not in group_by]
        if config_cols:
            # Use the first configuration column by default
            compare_by = config_cols[0]
        else:
            # Create a dummy column if no configuration columns exist
            df['_config'] = 'All Data'
            compare_by = '_config'
    
    # Convert compare_by to list if it's a string
    if isinstance(compare_by, str):
        compare_by = [compare_by]
    
    # Create a new column combining all compare_by columns for easier handling
    if len(compare_by) > 1:
        # Combine multiple columns into a single identifier
        df['_config_combined'] = df.apply(
            lambda row: f"{row['scale'].capitalize()} {'preprocessed ' if row['preprocessing'] == 'preprocessed' else ''}{row['data_type']}",
            axis=1
        )
        compare_column = '_config_combined'
    else:
        compare_column = compare_by[0]
    
    # Get unique groups and configs
    if len(group_by) > 1:
        # Create a combined group column
        df['_group_combined'] = df.apply(
            lambda row: f"{row['scale'].capitalize()} {'preprocessed ' if row['preprocessing'] == 'preprocessed' else ''}{row['data_type']}",
            axis=1
        )
        group_column = '_group_combined'
    else:
        group_column = group_by[0]
    
    unique_groups = df[group_column].unique()
    unique_configs = df[compare_column].unique()
    
    # Sort unique groups if requested and if they're metrics
    if sort_metrics and group_column == 'metric':
        unique_groups = sorted(unique_groups)
    
    # Set up colors
    if colors is None:
        # Generate colors using a colormap
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(len(unique_configs))]
    else:
        # Use provided colors, cycling if needed
        color_cycle = cycle(colors)
        colors = [next(color_cycle) for _ in range(len(unique_configs))]
    
    # Lighter colors for the bars, darker for the error bars
    bar_colors = [to_rgba(color, 0.7) for color in colors]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set width of a group
    group_width = bar_width
    # Width of an individual bar
    width = group_width / len(unique_configs)
    
    # Position of groups on x-axis
    group_positions = np.arange(len(unique_groups))
    
    # Plot bars for each configuration
    for i, config in enumerate(unique_configs):
        # Filter data for this configuration
        config_data = df[df[compare_column] == config]
        
        # Values for each group in this configuration
        values = []
        lower_errors = []
        upper_errors = []
        positions = []
        
        for j, group in enumerate(unique_groups):
            # Get data for this group and config
            group_data = config_data[config_data[group_column] == group]
            
            if not group_data.empty:
                # Append values and positions
                values.append(group_data['value'].iloc[0])
                lower_errors.append(group_data['value'].iloc[0] - group_data['lower_bound'].iloc[0])
                upper_errors.append(group_data['upper_bound'].iloc[0] - group_data['value'].iloc[0])
                positions.append(j)
            
        # Position adjustment for this configuration within each group
        position_adj = width * (i - len(unique_configs) / 2 + 0.5)
        bar_positions = [pos + position_adj for pos in positions]
        
        # Plot bars
        bars = ax.bar(
            bar_positions, 
            values, 
            width=width * 0.9,  # Slightly smaller than width for separation
            color=bar_colors[i],
            yerr=[lower_errors, upper_errors],
            capsize=4,
            label=config
        )
    
    # Set x-axis ticks and labels
    ax.set_xticks(group_positions)
    ax.set_xticklabels(unique_groups, rotation=45 if len(unique_groups) > 3 else 0, 
                       ha='right' if len(unique_groups) > 3 else 'center')
    
    # Set labels and title
    ax.set_xlabel(', '.join(group_by) if group_by != [x] else x, fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    if title is None:
        compare_str = ', '.join(compare_by)
        title = f"Metrics Comparison by {compare_str}"
    
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(title=', '.join(compare_by), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(f'../reports/figures/metrics_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()

def multi_model_facet(data, metrics=None, facet_by=None, compare_by=None,
                          figsize=(15, 10), title=None, colors=None):
    """
    Create a facet grid of plots showing metrics for multiple model configurations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing columns for metric, value, lower_bound, upper_bound,
        and additional columns for model configurations.
    
    metrics : list, optional
        List of metrics to include. If None, all metrics in the data are used.
    
    facet_by : str or list, optional
        Column(s) to create facets by. Default is None (single facet).
    
    compare_by : str or list, optional
        Column(s) to compare within each facet. Default is None (all data together).
    
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (15, 10).
    
    title : str, optional
        Main title for the figure. If None, a title will be generated.
    
    colors : list, optional
        List of colors for different configurations. If None, default colors are used.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    # Convert to DataFrame if it's a dict
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Filter metrics if specified
    if metrics is not None:
        df = df[df['metric'].isin(metrics)]
    
    # Set default facet_by if not provided
    if facet_by is None:
        # Find configuration columns
        config_cols = [col for col in df.columns 
                      if col not in ['metric', 'value', 'lower_bound', 'upper_bound']]
        if config_cols:
            # Use first configuration column by default
            facet_by = config_cols[0]
        else:
            # Create a dummy column if no configuration columns exist
            df['_facet'] = 'All Data'
            facet_by = '_facet'
    
    # Convert single column names to lists for consistent handling
    if isinstance(facet_by, str):
        facet_by = [facet_by]
    
    # Find compare_by columns if not specified
    if compare_by is None:
        # Find remaining configuration columns
        config_cols = [col for col in df.columns 
                      if col not in ['metric', 'value', 'lower_bound', 'upper_bound']
                      and col not in facet_by]
        if config_cols:
            # Use first remaining configuration column
            compare_by = config_cols[0]
        else:
            # Create a dummy column if no remaining configuration columns
            df['_compare'] = 'All Data'
            compare_by = '_compare'
    
    # Convert compare_by to list if it's a string
    if isinstance(compare_by, str):
        compare_by = [compare_by]
    
    # Create combined columns for faceting and comparison
    if len(facet_by) > 1:
        df['_facet_combined'] = df.apply(
            lambda row: f"{row['scale'].capitalize()} {'preprocessed ' if row['preprocessing'] == 'preprocessed' else ''}{row['data_type']}",
            axis=1
        )
        facet_column = '_facet_combined'
    else:
        facet_column = facet_by[0]
    
    if len(compare_by) > 1:
        df['_compare_combined'] = df.apply(
            lambda row: f"{row['scale'].capitalize()} {'preprocessed ' if row['preprocessing'] == 'preprocessed' else ''}{row['data_type']}",
            axis=1
        )
        compare_column = '_compare_combined'
    else:
        compare_column = compare_by[0]
    
    # Get unique metrics, facets, and comparison values
    unique_metrics = df['metric'].unique()
    unique_facets = df[facet_column].unique()
    unique_compares = df[compare_column].unique()
    
    # Set up colors
    if colors is None:
        # Generate colors using a colormap
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(len(unique_compares))]
    else:
        # Use provided colors, cycling if needed
        color_cycle = cycle(colors)
        colors = [next(color_cycle) for _ in range(len(unique_compares))]
    
    # Determine grid dimensions
    n_facets = len(unique_facets)
    n_rows = int(np.ceil(n_facets / 2))  # Max 2 columns
    n_cols = min(2, n_facets)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                           sharex=True, sharey=True)
    
    # Convert to 2D array for consistent indexing
    if n_facets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Set main title if provided
    if title is None:
        facet_str = ', '.join(facet_by)
        compare_str = ', '.join(compare_by)
        title = f"Metrics Comparison - Faceted by {facet_str}, Compared by {compare_str}"
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Plot each facet
    for i, facet in enumerate(unique_facets):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        
        # Filter data for this facet
        facet_data = df[df[facet_column] == facet]
        
        # Set up x positions
        metric_positions = np.arange(len(unique_metrics))
        width = 0.8 / len(unique_compares)
        
        # Plot each comparison group
        for j, comp in enumerate(unique_compares):
            # Filter data for this comparison group
            comp_data = facet_data[facet_data[compare_column] == comp]
            
            # Prepare data arrays
            values = []
            lower_errors = []
            upper_errors = []
            positions = []
            
            for k, metric in enumerate(unique_metrics):
                # Get data for this metric and comparison
                metric_data = comp_data[comp_data['metric'] == metric]
                
                if not metric_data.empty:
                    # Append values and positions
                    values.append(metric_data['value'].iloc[0])
                    lower_errors.append(metric_data['value'].iloc[0] - metric_data['lower_bound'].iloc[0])
                    upper_errors.append(metric_data['upper_bound'].iloc[0] - metric_data['value'].iloc[0])
                    positions.append(k)
            
            # Position adjustment for this comparison within each metric
            position_adj = width * (j - len(unique_compares) / 2 + 0.5)
            bar_positions = [pos + position_adj for pos in positions]
            
            # Plot bars
            ax.bar(
                bar_positions, 
                values, 
                width=width * 0.9,
                color=to_rgba(colors[j], 0.7),
                yerr=[lower_errors, upper_errors],
                capsize=3,
                label=comp if i == 0 else ""  # Only add label in first facet
            )
        
        # Set x-axis ticks and labels
        ax.set_xticks(metric_positions)
        ax.set_xticklabels(unique_metrics, rotation=45, ha='right')
        
        # Set facet title
        ax.set_title(f"{facet_column}: {facet}", fontsize=12)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Only add y-label to leftmost axes
        if col_idx == 0:
            ax.set_ylabel('Value', fontsize=12)
    
    # Hide empty subplots if any
    for i in range(n_facets, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].set_visible(False)
    
    # Add a single legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title=', '.join(compare_by), 
             loc='upper center', bbox_to_anchor=(0.5, 0.96),
             ncol=min(5, len(unique_compares)))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Make room for suptitle and legend
    
    # Save figure if path is provided
    plt.savefig(f'../reports/figures/metrics_confidence_intervals_facet.png', dpi=300, bbox_inches='tight')
    plt.show()
