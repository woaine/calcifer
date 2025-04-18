import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

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

def training_loss_r2(df, scale, preprocessing, data_type, type='validation', loss_zoom_range=None, r2_zoom_range=None):
    """
    Plot either training or validation loss alongside R2 scores with zoomed y-axis versions
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns 'loss', 'val_loss', 'r2_score', 'val_r2_score'
    data_type : str, optional
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
    
    # Validate data_type parameter
    if data_type.lower() not in ['training', 'validation']:
        raise ValueError("data_type must be either 'training' or 'validation'")
    
    # Set column names based on data_type
    if data_type.lower() == 'training':
        loss_col = 'loss'
        r2_col = 'r2_score'
        title_prefix = 'Training'
    else:  # validation
        loss_col = 'val_loss'
        r2_col = 'val_r2_score'
        title_prefix = 'Validation'
    
    color = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'olive']
    
    # Create a figure with four subplots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, 
                           gridspec_kw={'height_ratios': [1, 3]})
    
    # Generate epoch numbers (1-based)
    epochs = np.arange(1, len(df) + 1)
    
    # Calculate default zoom ranges if not provided
    if loss_zoom_range is None:
        # Calculate a reasonable y-axis zoom range for loss
        min_loss = df[loss_col].min()
        median_loss = df[loss_col].median()
        
        # Create a range with some padding
        loss_range = median_loss - min_loss
        loss_zoom_range = (
            max(0, min_loss - 0.1 * loss_range),  # Lower bound (ensure positive)
            min_loss + loss_range * 1.5  # Upper bound with padding
        )
    
    if r2_zoom_range is None:
        # Calculate a reasonable y-axis zoom range for R2
        min_r2 = df[r2_col].min()
        max_r2 = df[r2_col].max()
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
    
    # Plot the loss (top left)
    axs[0, 0].plot(epochs, df[loss_col], color=color, label=f'{title_prefix} Loss')
    axs[0, 0].set_title(f'{title_prefix} Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot the R2 scores (top right)
    axs[0, 1].plot(epochs, df[r2_col], color=color, label=f'{title_prefix} R2 Score')
    axs[0, 1].set_title(f'{title_prefix} R2 Score')
    axs[0, 1].set_ylabel('R2 Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot the loss with zoomed y-axis (bottom left)
    axs[1, 0].plot(epochs, df[loss_col], color=color, label=f'{title_prefix} Loss')

    # Set the zoomed y-range for loss
    axs[1, 0].set_ylim(loss_zoom_range)
    
    axs[1, 0].set_title(f'Zoomed Y-Axis: Loss Range [{loss_zoom_range[0]:.4f}, {loss_zoom_range[1]:.4f}]')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot the R2 scores with zoomed y-axis (bottom right)
    axs[1, 1].plot(epochs, df[r2_col], color=color, label=f'{title_prefix} R2 Score')

    # Set the zoomed y-range for R2
    axs[1, 1].set_ylim(r2_zoom_range)
    
    axs[1, 1].set_title(f'Zoomed Y-Axis: R2 Range [{r2_zoom_range[0]:.4f}, {r2_zoom_range[1]:.4f}]')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('R2 Score')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Save the figure automatically to the specified directory
    plt.savefig(f'../reports/figures/{type}_{scale}_{preprocessing}_{data_type}_training_progress.png', dpi=300, bbox_inches='tight') 

    # Add an overall title
    plt.suptitle(f'{title_prefix} Metrics', fontsize=16, y=0.98)
    
    # Adjust the layout with more space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
    plt.show()