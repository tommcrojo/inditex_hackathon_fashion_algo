"""
Feature processing components for recommender system
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


class FeatureProcessor:
    """Enhanced feature processor for recommendation data"""
    
    def __init__(self, session_sequence_length: int = 4):
        self.session_sequence_length = session_sequence_length
        self.scaler = StandardScaler()
        
    def process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp"""
        df = df.copy()
        
        # Use the timestamp column that should already be converted to datetime
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_local'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day (morning, afternoon, evening, night)
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Month and day for seasonal patterns
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        return df
    
    def process_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate session-based features"""
        df = df.copy()
        
        # Use the timestamp column that should already be converted to datetime
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_local'])
        
        # Sort by session and timestamp
        df = df.sort_values(['session_id', 'timestamp'])
        
        # Add sequence position within session
        df['seq_position'] = df.groupby('session_id').cumcount() + 1
        
        # Calculate session length (number of interactions)
        df['session_length'] = df.groupby('session_id')['seq_position'].transform('max')
        
        # Calculate time deltas between interactions
        df['time_delta'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds()
        
        # Calculate session duration
        session_duration = df.groupby('session_id').apply(
            lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds()
            if len(x) > 1 else 0
        ).reset_index()
        session_duration.columns = ['session_id', 'session_duration']
        
        df = df.merge(session_duration, on='session_id', how='left')
        
        # Calculate average time between interactions
        session_avg_time = df.groupby('session_id')['time_delta'].mean().reset_index()
        session_avg_time.columns = ['session_id', 'avg_interaction_time']
        
        df = df.merge(session_avg_time, on='session_id', how='left')
        
        # Fill NaN values for first interactions
        df['time_delta'] = df['time_delta'].fillna(0)
        df['avg_interaction_time'] = df['avg_interaction_time'].fillna(0)
        
        # Flag if user added anything to cart in this session
        if 'add_to_cart' in df.columns:
            cart_sessions = df[df['add_to_cart'] == 1]['session_id'].unique()
            df['has_cart_addition'] = df['session_id'].isin(cart_sessions).astype(int)
        
        return df
    
    def process_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user profile features"""
        # Ensure timestamp is datetime format for faster processing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_local'])
        
        # Calculate basic RFM metrics from interactions
        latest_date = df['timestamp'].max()
        
        # Group by user_id and calculate metrics
        user_rfm = df.groupby('user_id').agg({
            'timestamp': lambda x: (latest_date - x.max()).days,  # Recency (days)
            'session_id': 'nunique',  # Frequency (sessions)
            'add_to_cart': 'sum'  # Monetary (cart additions as proxy)
        }).reset_index()
        
        user_rfm.columns = ['user_id', 'R', 'F', 'M']
        
        # Invert R so higher is better
        max_r = user_rfm['R'].max()
        user_rfm['R'] = max_r - user_rfm['R']
        
        # Scale RFM values
        if len(user_rfm) > 1:
            user_rfm[['R', 'F', 'M']] = self.scaler.fit_transform(user_rfm[['R', 'F', 'M']])
        
        # Add country preference (most frequent country)
        user_country = df.groupby(['user_id', 'country']).size().reset_index()
        user_country.columns = ['user_id', 'country', 'count']
        user_country = user_country.sort_values(['user_id', 'count'], ascending=[True, False])
        user_country = user_country.drop_duplicates('user_id')
        user_country = user_country[['user_id', 'country']]
        user_country.columns = ['user_id', 'preferred_country']
        
        # Add device preference
        user_device = df.groupby(['user_id', 'device_type']).size().reset_index()
        user_device.columns = ['user_id', 'device_type', 'count']
        user_device = user_device.sort_values(['user_id', 'count'], ascending=[True, False])
        user_device = user_device.drop_duplicates('user_id')
        user_device = user_device[['user_id', 'device_type']]
        user_device.columns = ['user_id', 'preferred_device']
        
        # Merge all user features
        user_features = user_rfm
        user_features = user_features.merge(user_country, on='user_id', how='left')
        user_features = user_features.merge(user_device, on='user_id', how='left')
        
        return user_features
    
    def process_product_features(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Process product features for content-based recommendations"""
        products_df = products_df.copy()
        
        # Normalize categorical features
        if 'familiy' in products_df.columns:
            products_df['family_normalized'] = products_df['familiy'] / products_df['familiy'].max()
        
        if 'cod_section' in products_df.columns:
            products_df['section_normalized'] = products_df['cod_section'] / products_df['cod_section'].max()
        
        if 'color_id' in products_df.columns:
            products_df['color_normalized'] = products_df['color_id'] / products_df['color_id'].max()
        
        # Create discount flag
        if 'discount' not in products_df.columns:
            products_df['discount'] = False
        
        # Create normalized embeddings
        if 'embedding' in products_df.columns:
            # Check if embeddings need normalization
            sample_embedding = products_df.iloc[0]['embedding']
            if isinstance(sample_embedding, (list, np.ndarray)) and len(sample_embedding) > 0:
                # Normalize embeddings to unit length
                products_df['embedding_norm'] = products_df['embedding'].apply(
                    lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x
                )
        
        return products_df
    
    def calculate_product_similarity(self, products_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate product similarity based on embeddings"""
        if 'embedding' not in products_df.columns:
            return {}
        
        # Extract products with valid embeddings
        valid_products = products_df[products_df['embedding'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )]
        
        if len(valid_products) < 2:
            return {}
        
        # Create a dict mapping product IDs to their embeddings
        product_embeddings = {}
        for _, row in valid_products.iterrows():
            product_embeddings[row['partnumber']] = row['embedding']
        
        # Calculate similarity matrix
        product_ids = list(product_embeddings.keys())
        embedding_matrix = np.array(list(product_embeddings.values()))
        
        # Normalize embeddings
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embedding_matrix = embedding_matrix / norms
        
        # Calculate cosine similarity
        similarity_matrix = embedding_matrix @ embedding_matrix.T
        
        # Create similarity dictionary
        similarity_dict = {}
        max_similar = 50  # Limit to top 50 similar products for each product
        
        for i, prod_id in enumerate(product_ids):
            # Get indices of most similar products (excluding self)
            similar_indices = np.argsort(-similarity_matrix[i])
            similar_indices = similar_indices[similar_indices != i][:max_similar]
            
            # Map back to product IDs with similarity scores
            similarity_dict[prod_id] = [
                (product_ids[idx], float(similarity_matrix[i, idx]))
                for idx in similar_indices
            ]
        
        return similarity_dict
    
    def get_popular_products(self, train_df: pd.DataFrame, 
                             by_country: bool = True, 
                             by_device: bool = False) -> Dict[Any, List[str]]:
        """Get popular products, optionally by country or device"""
        # Base popularity (products added to cart)
        cart_products = train_df[train_df['add_to_cart'] == 1]
        
        if by_country and not by_device:
            # Group by country
            popular_by_country = {}
            for country in cart_products['country'].unique():
                country_products = cart_products[cart_products['country'] == country]
                popular = country_products['partnumber'].value_counts().head(20).index.tolist()
                popular_by_country[country] = popular
            return popular_by_country
        
        elif by_device and not by_country:
            # Group by device
            popular_by_device = {}
            for device in cart_products['device_type'].unique():
                device_products = cart_products[cart_products['device_type'] == device]
                popular = device_products['partnumber'].value_counts().head(20).index.tolist()
                popular_by_device[device] = popular
            return popular_by_device
        
        elif by_country and by_device:
            # Group by country and device
            popular_by_country_device = {}
            for country in cart_products['country'].unique():
                for device in cart_products['device_type'].unique():
                    country_device_products = cart_products[
                        (cart_products['country'] == country) & 
                        (cart_products['device_type'] == device)
                    ]
                    if len(country_device_products) > 0:
                        popular = country_device_products['partnumber'].value_counts().head(20).index.tolist()
                        popular_by_country_device[(country, device)] = popular
            return popular_by_country_device
        
        else:
            # Overall popularity
            popular = cart_products['partnumber'].value_counts().head(50).index.tolist()
            return {'overall': popular}
    
    def prepare_all_features(self, train_df: pd.DataFrame, 
                             products_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process all features and return dictionary of feature dataframes"""
        # Make a copy to avoid modifying original
        df_copy = train_df.copy()
        
        # Convert timestamp_local to datetime once for efficiency
        print("Converting timestamps...")
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp_local'])
        
        print("Processing temporal features...")
        temporal_df = self.process_temporal_features(df_copy)
        
        print("Processing session features...")
        session_df = self.process_session_features(temporal_df)
        
        print("Processing user features...")
        user_df = self.process_user_features(session_df)
        
        print("Processing product features...")
        product_df = self.process_product_features(products_df)
        
        print("Calculating product similarities...")
        product_similarities = self.calculate_product_similarity(product_df)
        
        print("Getting popular products...")
        popular_products = self.get_popular_products(train_df, by_country=True, by_device=True)
        
        return {
            'processed_interactions': session_df,
            'user_features': user_df,
            'product_features': product_df,
            'product_similarities': product_similarities,
            'popular_products': popular_products
        }