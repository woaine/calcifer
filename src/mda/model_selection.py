import pandas as pd

def filter_models(df: pd.DataFrame):
    """
    Filters and selects the best-performing epochs for each unique model configuration 
    in a given DataFrame based on validation loss and the gap between training and 
    validation loss.
    This function processes a DataFrame containing model training results, where each 
    row corresponds to a specific epoch of training for a particular model configuration. 
    It identifies the best epoch for each unique combination of `layers`, `activation`, 
    and `batch_size` by calculating a combined score based on normalized validation loss 
    and the absolute difference between training and validation loss (loss gap). The 
    epoch with the lowest combined score is selected as the best epoch for that model 
    configuration.
    Args:
        df (pd.DataFrame): A DataFrame containing model training results. It must include 
            the following columns:
            - 'layers': The number of layers in the model.
            - 'activation': The activation function used in the model.
            - 'batch_size': The batch size used during training.
            - 'loss': The training loss for the epoch.
            - 'val_loss': The validation loss for the epoch.
    Returns:
        pd.DataFrame: A DataFrame containing the best epoch for each unique model 
        configuration. The returned DataFrame includes all original columns from the 
        input DataFrame, except for the temporary columns used during processing 
        ('loss_gap', 'norm_val_loss', 'norm_loss_gap', 'score').
    Notes:
        - If there is only one epoch for a given model configuration, normalization 
          is skipped, and the single epoch is selected as the best epoch.
        - The combined score is calculated as the sum of the normalized validation 
          loss and the normalized loss gap, giving equal weight to both metrics.
    """
    
    # Get unique model configurations
    model_configs = df[['layers', 'activation', 'batch_size']].drop_duplicates()
    
    # Store results
    best_epochs = []
    
    # Process each model configuration
    for _, config in model_configs.iterrows():
        # Filter data for this specific model configuration
        model_df = df[(df['layers'] == config['layers']) & 
                     (df['activation'] == config['activation']) & 
                     (df['batch_size'] == config['batch_size'])].copy()
        
        # Calculate the absolute difference between training and validation loss
        model_df['loss_gap'] = abs(model_df['loss'] - model_df['val_loss'])
        
        # Normalize both metrics to [0,1] scale for equal weighting
        if len(model_df) > 1:
            # Normalize validation loss
            val_loss_min = model_df['val_loss'].min()
            val_loss_max = model_df['val_loss'].max()
            val_loss_range = val_loss_max - val_loss_min
            
            if val_loss_range > 0:
                model_df['norm_val_loss'] = (model_df['val_loss'] - val_loss_min) / val_loss_range
            else:
                model_df['norm_val_loss'] = 0
                
            # Normalize loss gap
            gap_min = model_df['loss_gap'].min()
            gap_max = model_df['loss_gap'].max()
            gap_range = gap_max - gap_min
            
            if gap_range > 0:
                model_df['norm_loss_gap'] = (model_df['loss_gap'] - gap_min) / gap_range
            else:
                model_df['norm_loss_gap'] = 0
        else:
            # If there's only one epoch, normalization is not needed
            model_df['norm_val_loss'] = 0
            model_df['norm_loss_gap'] = 0
        
        # Calculate combined score (equal weights by default)
        model_df['score'] = model_df['norm_val_loss'] + model_df['norm_loss_gap']
        
        # Find the epoch with the minimum score
        best_epoch_idx = model_df['score'].idxmin()
        best_epoch_row = model_df.loc[best_epoch_idx]
        
        # Add to our results
        best_epochs.append(best_epoch_row)
    
    # Create the result DataFrame
    result_df = pd.DataFrame(best_epochs)
    
    # Remove the temporary columns we added
    if not result_df.empty:
        result_df = result_df.drop(['loss_gap', 'norm_val_loss', 'norm_loss_gap', 'score'], axis=1)
    
    return result_df