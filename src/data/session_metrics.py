"""
Task 2 implementation: get_session_metrics function
"""
import pandas as pd


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Generate session metrics including total session time and cart addition ratio
    
    Parameters:
    df (pd.DataFrame): DataFrame with the format of the TRAIN dataset
    user_id (int): User identifier to filter data
    
    Returns:
    pd.DataFrame: DataFrame with columns ["user_id", "session_id", "total_session_time", "cart_addition_ratio"]
    """
    # Filter for specified user
    user_df = df[df['user_id'] == user_id].copy()
    
    if user_df.empty:
        return pd.DataFrame(columns=["user_id", "session_id", "total_session_time", "cart_addition_ratio"])
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(user_df['timestamp_local']):
        user_df['timestamp'] = pd.to_datetime(user_df['timestamp_local'])
    else:
        user_df['timestamp'] = user_df['timestamp_local']
    
    # Calculate session duration
    session_metrics = []
    
    for session_id, session_df in user_df.groupby('session_id'):
        # Sort by timestamp
        session_df = session_df.sort_values('timestamp')
        
        # Calculate total session time in seconds
        if len(session_df) > 1:
            total_session_time = (session_df['timestamp'].max() - session_df['timestamp'].min()).total_seconds()
        else:
            total_session_time = 0
        
        # Calculate cart addition ratio
        total_products = session_df['partnumber'].nunique()
        cart_products = session_df[session_df['add_to_cart'] == 1]['partnumber'].nunique()
        
        cart_addition_ratio = (cart_products / total_products * 100) if total_products > 0 else 0
        
        session_metrics.append({
            'user_id': user_id,
            'session_id': session_id,
            'total_session_time': total_session_time,
            'cart_addition_ratio': cart_addition_ratio
        })
    
    # Create DataFrame and sort by user_id and session_id
    result_df = pd.DataFrame(session_metrics)
    
    if not result_df.empty:
        result_df = result_df.sort_values(['user_id', 'session_id'])
    
    return result_df