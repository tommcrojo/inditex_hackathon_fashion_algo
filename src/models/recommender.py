"""
Recommender system implementations including collaborative, content-based, 
contextual, and hybrid recommendation approaches
"""
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


class RecommendationDataset(Dataset):
    """Dataset for training recommendation models"""
    
    def __init__(self, interactions, user_mapping, product_mapping, include_features=False, products_df=None):
        self.interactions = interactions
        self.user_mapping = user_mapping
        self.product_mapping = product_mapping
        self.include_features = include_features
        self.products_df = products_df
        
        # Prepare additional features if requested
        if self.include_features and self.products_df is not None:
            # Create features for each product
            self.product_features = {}
            for _, row in self.products_df.iterrows():
                product_id = row['partnumber']
                # Simple features: discount (0/1), family and section
                features = [
                    int(row.get('discount', False)),
                ]
                # Add family if exists
                if 'familiy' in row:
                    features.append(row['familiy'] / 1000)  # Normalize
                elif 'family' in row:
                    features.append(row['family'] / 1000)  # Normalize
                else:
                    features.append(0)
                
                # Add section if exists
                if 'cod_section' in row:
                    features.append(row['cod_section'] / 100)  # Normalize
                else:
                    features.append(0)
                
                self.product_features[product_id] = torch.tensor(features, dtype=torch.float)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id = self.interactions.iloc[idx]['user_id']
        product_id = self.interactions.iloc[idx]['partnumber']
        label = float(self.interactions.iloc[idx]['add_to_cart'])
        
        # Convert to tensor indices
        user_idx = self.user_mapping.get(user_id, 0)  # 0 for unknown users
        product_idx = self.product_mapping.get(product_id, 0)  # 0 for unknown products
        
        if self.include_features and self.products_df is not None and product_id in self.product_features:
            prod_features = self.product_features[product_id]
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'product_idx': torch.tensor(product_idx, dtype=torch.long),
                'product_features': prod_features,
                'label': torch.tensor(label, dtype=torch.float)
            }
        else:
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'product_idx': torch.tensor(product_idx, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }


class ProductEmbeddingModel(nn.Module):
    """Neural network model for collaborative filtering with product features"""
    
    def __init__(self, num_users, num_products, embedding_dim=64, feature_dim=3, hidden_dims=[128, 64]):
        super(ProductEmbeddingModel, self).__init__()
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)  # +1 for unknown user
        self.product_embedding = nn.Embedding(num_products + 1, embedding_dim)  # +1 for unknown product
        
        # Combined input: user embedding + product embedding + product features
        combined_dim = embedding_dim * 2 + feature_dim
        
        # Prediction layers
        layers = []
        prev_dim = combined_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, user_idx, product_idx, product_features=None):
        # Get embeddings
        user_embeds = self.user_embedding(user_idx)
        product_embeds = self.product_embedding(product_idx)
        
        # Concatenate embeddings and features if available
        if product_features is not None:
            concat_embeds = torch.cat([user_embeds, product_embeds, product_features], dim=1)
        else:
            concat_embeds = torch.cat([user_embeds, product_embeds], dim=1)
        
        # Generate prediction
        output = self.fc_layers(concat_embeds)
        return output.squeeze()


class CollaborativeFilteringModel:
    """Collaborative filtering model with temporal decay"""
    
    def __init__(self, embedding_dim: int = 64, feature_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.model = None
        self.user_mapping = {}
        self.product_mapping = {}
        self.idx_to_product = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, train_df: pd.DataFrame, products_df: pd.DataFrame, 
                    test_size: float = 0.2, balance_ratio: int = 3):
        """Prepare training data with balanced positive/negative samples"""
        self.train_df = train_df.copy()
        self.products_df = products_df.copy()
        
        # Create user and product mappings
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        
        # Ensure user_id is string for consistency
        self.train_df['user_id'] = self.train_df['user_id'].astype(str)
        
        # Encode users
        unique_users = self.train_df['user_id'].dropna().unique()
        self.user_encoder.fit(unique_users)
        
        # Encode products
        unique_products = self.products_df['partnumber'].unique()
        self.product_encoder.fit(unique_products)
        
        # Create mappings for fast lookup
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(self.user_encoder.classes_)}
        self.product_mapping = {prod_id: idx for idx, prod_id in enumerate(self.product_encoder.classes_)}
        
        # Reverse mapping for products
        self.idx_to_product = {idx: prod_id for prod_id, idx in self.product_mapping.items()}
        
        # Balance the dataset
        positive_samples = self.train_df[self.train_df['add_to_cart'] == 1]
        negative_samples = self.train_df[self.train_df['add_to_cart'] == 0]
        
        # Downsample negative samples
        sample_size = min(len(positive_samples) * balance_ratio, len(negative_samples))
        negative_samples = negative_samples.sample(sample_size, random_state=42)
        
        # Combine datasets
        balanced_df = pd.concat([positive_samples, negative_samples])
        
        # Split into train and validation
        train_interactions, val_interactions = train_test_split(
            balanced_df, test_size=test_size, random_state=42, 
            stratify=balanced_df['add_to_cart']
        )
        
        # Create datasets
        train_dataset = self._create_dataset(train_interactions)
        val_dataset = self._create_dataset(val_interactions)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=True,
            num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=1024, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_dataset(self, interactions):
        """Create PyTorch dataset for training"""
        return RecommendationDataset(
            interactions, self.user_mapping, self.product_mapping,
            include_features=True, products_df=self.products_df
        )
    
    def build_model(self):
        """Build neural network model"""
        feature_dim = self.feature_dim  # discount, family, section
        self.model = ProductEmbeddingModel(
            len(self.user_mapping), 
            len(self.product_mapping),
            self.embedding_dim,
            feature_dim=feature_dim
        ).to(self.device)
        return self.model
    
    def train(self, train_df: pd.DataFrame, products_df: pd.DataFrame, 
              epochs: int = 5, learning_rate: float = 0.001, model_path: str = None):
        """Train the collaborative filtering model"""
        # Prepare data
        train_loader, val_loader = self.prepare_data(train_df, products_df)
        
        # Build model
        self.build_model()
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # Use learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        
        print(f"Training collaborative filtering model...")
        
        # Variables for tracking
        best_val_loss = float('inf')
        early_stop_count = 0
        early_stop_patience = 3
        
        # Create model directory if it doesn't exist
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            start_time = time.time()
            for batch in train_loader:
                # Move data to device
                user_idx = batch['user_idx'].to(self.device)
                product_idx = batch['product_idx'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get product features if available
                product_features = batch.get('product_features', None)
                if product_features is not None:
                    product_features = product_features.to(self.device)
                
                # Forward pass
                predictions = self.model(user_idx, product_idx, product_features)
                loss = criterion(predictions, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * len(labels)
                pred_labels = (predictions >= 0.5).float()
                train_correct += (pred_labels == labels).sum().item()
                train_total += len(labels)
            
            train_loss /= train_total
            train_accuracy = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move data to device
                    user_idx = batch['user_idx'].to(self.device)
                    product_idx = batch['product_idx'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Get product features if available
                    product_features = batch.get('product_features', None)
                    if product_features is not None:
                        product_features = product_features.to(self.device)
                    
                    # Forward pass
                    predictions = self.model(user_idx, product_idx, product_features)
                    loss = criterion(predictions, labels)
                    
                    # Statistics
                    val_loss += loss.item() * len(labels)
                    pred_labels = (predictions >= 0.5).float()
                    val_correct += (pred_labels == labels).sum().item()
                    val_total += len(labels)
            
            val_loss /= val_total
            val_accuracy = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_path:
                    torch.save(self.model.state_dict(), model_path)
                early_stop_count = 0
                print(f"  Saved best model with validation loss: {val_loss:.4f}")
            else:
                early_stop_count += 1
            
            # Show progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - "
                  f"Train loss: {train_loss:.4f}, acc: {train_accuracy:.4f} - "
                  f"Val loss: {val_loss:.4f}, acc: {val_accuracy:.4f}")
            
            # Early stopping
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model if saved
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            
        print("Training complete!")
        return self.model
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, user_id, n=5, exclude_products=None):
        """Generate predictions for a user"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before predictions")
        
        self.model.eval()
        
        # Convert user_id to index
        if user_id is not None and user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
        else:
            user_idx = 0  # Unknown user
        
        user_tensor = torch.tensor([user_idx]).to(self.device)
        
        # Products to exclude
        if exclude_products is None:
            exclude_products = []
        
        # Score products in batches
        scores = {}
        batch_size = 1000
        all_products = list(self.product_mapping.keys())
        
        for i in range(0, len(all_products), batch_size):
            batch_products = all_products[i:i+batch_size]
            
            # Filter out products to exclude
            batch_products = [p for p in batch_products if p not in exclude_products]
            
            if not batch_products:
                continue
            
            # Build indices and tensors
            product_indices = [self.product_mapping[p] for p in batch_products]
            product_tensors = torch.tensor(product_indices).to(self.device)
            
            # Expand user to match batch size
            users_expanded = user_tensor.expand(len(product_indices))
            
            # Prepare product features if needed
            product_features = None
            if hasattr(self.model, 'fc_layers') and self.model.fc_layers[0].in_features > self.embedding_dim * 2:
                # Gather features
                features_list = []
                for prod in batch_products:
                    product_row = self.products_df[self.products_df['partnumber'] == prod].iloc[0]
                    features = [
                        float(product_row.get('discount', False)),
                    ]
                    
                    # Add family if exists
                    if 'familiy' in product_row:
                        features.append(product_row['familiy'] / 1000)
                    elif 'family' in product_row:
                        features.append(product_row['family'] / 1000)
                    else:
                        features.append(0)
                    
                    # Add section if exists
                    if 'cod_section' in product_row:
                        features.append(product_row['cod_section'] / 100)
                    else:
                        features.append(0)
                    
                    features_list.append(features)
                
                product_features = torch.tensor(features_list, dtype=torch.float).to(self.device)
            
            # Predict scores
            with torch.no_grad():
                batch_scores = self.model(users_expanded, product_tensors, product_features).cpu().numpy()
            
            # Save scores
            for j, product_id in enumerate(batch_products):
                scores[product_id] = float(batch_scores[j])
        
        # Sort and return top n
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [prod_id for prod_id, _ in sorted_products[:n]]


class ContentBasedRecommender:
    """Content-based recommender using product attributes and embeddings"""
    
    def __init__(self, product_similarities=None, product_features=None):
        self.product_similarities = product_similarities or {}
        self.product_features = product_features
        
    def set_product_similarities(self, similarities):
        """Set product similarities dictionary"""
        self.product_similarities = similarities
    
    def set_product_features(self, product_features):
        """Set product features dataframe"""
        self.product_features = product_features
    
    def recommend_similar_products(self, product_id, n=5, exclude_products=None):
        """Recommend products similar to a given product"""
        if exclude_products is None:
            exclude_products = []
        
        # Check if product is in similarity dict
        if product_id not in self.product_similarities:
            return []
        
        # Get similar products excluding those in exclude list
        similar_products = self.product_similarities[product_id]
        filtered_products = [(p, s) for p, s in similar_products if p not in exclude_products]
        
        # Return top n
        top_n = [p for p, _ in filtered_products[:n]]
        return top_n
    
    def recommend_from_interactions(self, interaction_products, n=5, exclude_products=None):
        """Recommend products based on a list of interacted products"""
        if exclude_products is None:
            exclude_products = []
        
        # Initialize scores dictionary
        scores = defaultdict(float)
        
        # For each product in interactions, add similarity scores to other products
        for product_id in interaction_products:
            if product_id in self.product_similarities:
                for similar_prod, similarity in self.product_similarities[product_id]:
                    if similar_prod not in exclude_products:
                        scores[similar_prod] += similarity
        
        # Sort by score and return top n
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [prod_id for prod_id, _ in sorted_products[:n]]
    
    def recommend_by_attributes(self, attributes, n=5, exclude_products=None):
        """Recommend products based on specified attributes"""
        if self.product_features is None:
            return []
        
        if exclude_products is None:
            exclude_products = []
        
        # Filter products by attributes
        filtered_products = self.product_features.copy()
        
        for attr, value in attributes.items():
            if attr in filtered_products.columns:
                filtered_products = filtered_products[filtered_products[attr] == value]
        
        # Exclude products
        filtered_products = filtered_products[~filtered_products['partnumber'].isin(exclude_products)]
        
        # Return top n by some criteria (e.g., discount first)
        if 'discount' in filtered_products.columns:
            sorted_products = filtered_products.sort_values(['discount'], ascending=False)
        else:
            sorted_products = filtered_products
        
        return sorted_products['partnumber'].head(n).tolist()


class ContextualRecommender:
    """Context-aware recommender that considers session, device, and time"""
    
    def __init__(self, train_df=None, popular_products=None):
        self.train_df = train_df
        self.popular_products = popular_products or {}
    
    def set_train_data(self, train_df):
        """Set training data"""
        self.train_df = train_df
    
    def set_popular_products(self, popular_products):
        """Set popular products dictionary"""
        self.popular_products = popular_products
    
    def recommend_by_context(self, context, n=5, exclude_products=None):
        """Recommend products based on contextual information"""
        if exclude_products is None:
            exclude_products = []
        
        country = context.get('country')
        device_type = context.get('device_type')
        
        # Try to get country and device specific recommendations
        if country is not None and device_type is not None and (country, device_type) in self.popular_products:
            popular = self.popular_products[(country, device_type)]
        # Fall back to country-specific
        elif country is not None and country in self.popular_products:
            popular = self.popular_products[country]
        # Fall back to device-specific
        elif device_type is not None and device_type in self.popular_products:
            popular = self.popular_products[device_type]
        # Fall back to overall popularity
        elif 'overall' in self.popular_products:
            popular = self.popular_products['overall']
        else:
            return []
        
        # Filter out excluded products
        filtered_popular = [p for p in popular if p not in exclude_products]
        
        return filtered_popular[:n]
    
    def recommend_by_time(self, hour_of_day, day_of_week, n=5, exclude_products=None):
        """Recommend products based on time patterns"""
        if self.train_df is None:
            return []
        
        if exclude_products is None:
            exclude_products = []
        
        # Create temporal features if not already present
        if 'hour' not in self.train_df.columns:
            self.train_df['timestamp'] = pd.to_datetime(self.train_df['timestamp_local'])
            self.train_df['hour'] = self.train_df['timestamp'].dt.hour
            self.train_df['day_of_week'] = self.train_df['timestamp'].dt.dayofweek
        
        # Filter for cart additions at similar time
        time_filtered = self.train_df[
            (self.train_df['hour'] == hour_of_day) & 
            (self.train_df['day_of_week'] == day_of_week) & 
            (self.train_df['add_to_cart'] == 1)
        ]
        
        # Get popular products for this time
        popular_at_time = time_filtered['partnumber'].value_counts().head(20).index.tolist()
        
        # Filter out excluded products
        filtered_popular = [p for p in popular_at_time if p not in exclude_products]
        
        return filtered_popular[:n]


class HybridRecommender:
    """Hybrid recommender system combining multiple recommendation strategies"""
    
    def __init__(self, data_directory=None, output_directory=None):
        self.data_directory = data_directory or '.'
        self.output_directory = output_directory or '.'
        
        # Create output directory if it doesn't exist
        if self.output_directory:
            os.makedirs(self.output_directory, exist_ok=True)
        
        # Create models directory
        self.model_directory = os.path.join(self.data_directory, 'models')
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Individual recommender components
        self.cf_model = CollaborativeFilteringModel()
        self.content_model = ContentBasedRecommender()
        self.context_model = ContextualRecommender()
        
        # Data
        self.train_df = None
        self.products_df = None
        self.test_df = None
        self.features = None
        
        # Model paths
        self.cf_model_path = os.path.join(self.model_directory, 'collaborative_model.pt')
    
    def load_data(self, train_path=None, products_path=None, test_path=None):
        """Load data from files"""
        print("Loading data...")
        
        # Load train data
        if train_path:
            self.train_df = pd.read_csv(train_path)
            print(f"Train data loaded: {self.train_df.shape}")
        
        # Load products data
        if products_path:
            import pickle
            with open(products_path, 'rb') as f:
                self.products_df = pickle.load(f)
            print(f"Products data loaded: {self.products_df.shape}")
        
        # Load test data
        if test_path:
            self.test_df = pd.read_csv(test_path)
            print(f"Test data loaded: {self.test_df.shape}")
    
    def process_features(self, feature_processor):
        """Process all features using the provided feature processor"""
        if self.train_df is None or self.products_df is None:
            raise ValueError("Data must be loaded before processing features")
        
        self.features = feature_processor.prepare_all_features(
            self.train_df, self.products_df
        )
        
        # Set processed data in component models
        self.content_model.set_product_similarities(self.features['product_similarities'])
        self.content_model.set_product_features(self.features['product_features'])
        self.context_model.set_train_data(self.features['processed_interactions'])
        self.context_model.set_popular_products(self.features['popular_products'])
    
    def train_models(self, train_cf=True):
        """Train all component models"""
        if self.features is None:
            raise ValueError("Features must be processed before training models")
        
        if train_cf:
            print("Training collaborative filtering model...")
            self.cf_model.train(
                self.features['processed_interactions'],
                self.features['product_features'],
                epochs=5,
                model_path=self.cf_model_path
            )
    
    def load_trained_models(self):
        """Load pre-trained models"""
        if os.path.exists(self.cf_model_path):
            print("Loading collaborative filtering model...")
            # Prepare mappings first
            self.cf_model.prepare_data(
                self.features['processed_interactions'],
                self.features['product_features']
            )
            self.cf_model.load_model(self.cf_model_path)
    
    def identify_user_type(self, user_id, session_data):
        """Identify user type (new/returning, with/without session data)"""
        is_new_user = user_id not in self.cf_model.user_mapping
        has_session_data = session_data is not None and not session_data.empty
        
        if is_new_user and not has_session_data:
            return "new_no_session"
        elif is_new_user and has_session_data:
            return "new_with_session"
        elif not is_new_user and not has_session_data:
            return "returning_no_session"
        else:
            return "returning_with_session"
    
    def recommend(self, session_id, n=5):
        """Generate recommendations for a session"""
        if self.test_df is None:
            raise ValueError("Test data must be loaded")
        
        # Get session data
        session_data = self.test_df[self.test_df['session_id'] == session_id]
        
        if session_data.empty:
            print(f"Warning: No data found for session {session_id}")
            return []
        
        # Get user_id
        user_id = session_data['user_id'].iloc[0]
        if pd.isna(user_id):
            user_id = None
        else:
            user_id = str(user_id)
        
        # Get context information
        context = {
            'country': session_data['country'].iloc[0],
            'device_type': session_data['device_type'].iloc[0],
        }
        
        # Add temporal context if available
        if 'timestamp_local' in session_data.columns:
            timestamp = pd.to_datetime(session_data['timestamp_local'].iloc[0])
            context['hour'] = timestamp.hour
            context['day_of_week'] = timestamp.dayofweek
        
        # Get viewed products in session
        viewed_products = session_data['partnumber'].unique().tolist() if not session_data.empty else []
        
        # Identify user type
        user_type = self.identify_user_type(user_id, session_data)
        
        # Generate recommendations based on user type
        if user_type == "new_no_session":
            return self._recommend_new_user_no_session(context, n)
        elif user_type == "new_with_session":
            return self._recommend_new_user_with_session(context, viewed_products, n)
        elif user_type == "returning_no_session":
            return self._recommend_returning_user_no_session(user_id, context, n)
        else:  # returning_with_session
            return self._recommend_returning_user_with_session(user_id, context, viewed_products, n)
    
    def _recommend_new_user_no_session(self, context, n=5):
        """Recommend for new user without session data"""
        # Use contextual recommendations
        contextual_recs = self.context_model.recommend_by_context(context, n=n)
        
        # If needed, fill with popular products
        if len(contextual_recs) < n and 'overall' in self.features['popular_products']:
            overall_popular = self.features['popular_products']['overall']
            for prod in overall_popular:
                if prod not in contextual_recs:
                    contextual_recs.append(prod)
                    if len(contextual_recs) >= n:
                        break
        
        return contextual_recs[:n]
    
    def _recommend_new_user_with_session(self, context, viewed_products, n=5):
        """Recommend for new user with session data"""
        # Get recommendations based on viewed products
        content_recs = self.content_model.recommend_from_interactions(
            viewed_products, n=n, exclude_products=viewed_products
        )
        
        # If not enough recommendations, fill with contextual recommendations
        if len(content_recs) < n:
            contextual_recs = self.context_model.recommend_by_context(
                context, n=n, exclude_products=viewed_products + content_recs
            )
            for prod in contextual_recs:
                if prod not in content_recs:
                    content_recs.append(prod)
                    if len(content_recs) >= n:
                        break
        
        return content_recs[:n]
    
    def _recommend_returning_user_no_session(self, user_id, context, n=5):
        """Recommend for returning user without session data"""
        # Use collaborative filtering
        cf_recs = self.cf_model.predict(user_id, n=n)
        
        # If not enough recommendations, fill with contextual recommendations
        if len(cf_recs) < n:
            contextual_recs = self.context_model.recommend_by_context(
                context, n=n, exclude_products=cf_recs
            )
            for prod in contextual_recs:
                if prod not in cf_recs:
                    cf_recs.append(prod)
                    if len(cf_recs) >= n:
                        break
        
        return cf_recs[:n]
    
    def _recommend_returning_user_with_session(self, user_id, context, viewed_products, n=5):
        """Recommend for returning user with session data"""
        # Combine collaborative filtering and content-based recommendations
        cf_weight = 0.7
        content_weight = 0.3
        
        # Get recommendations from each approach
        cf_recs = self.cf_model.predict(user_id, n=10, exclude_products=viewed_products)
        content_recs = self.content_model.recommend_from_interactions(
            viewed_products, n=10, exclude_products=viewed_products
        )
        
        # Combine with weighted voting
        scores = defaultdict(float)
        
        # Add CF scores
        for i, prod in enumerate(cf_recs):
            scores[prod] += cf_weight * (1.0 - i/len(cf_recs))
        
        # Add content-based scores
        for i, prod in enumerate(content_recs):
            scores[prod] += content_weight * (1.0 - i/len(content_recs))
        
        # Sort by combined score
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        combined_recs = [prod for prod, _ in sorted_products[:n]]
        
        # If not enough recommendations, add contextual
        if len(combined_recs) < n:
            contextual_recs = self.context_model.recommend_by_context(
                context, n=n, exclude_products=viewed_products + combined_recs
            )
            for prod in contextual_recs:
                if prod not in combined_recs:
                    combined_recs.append(prod)
                    if len(combined_recs) >= n:
                        break
        
        return combined_recs[:n]
    
    def generate_all_predictions(self, output_path=None):
        """Generate predictions for all test sessions"""
        if self.test_df is None:
            raise ValueError("Test data must be loaded")
        
        if output_path is None:
            output_path = os.path.join(self.output_directory, 'predictions_3.json')
        
        # Get all unique session IDs
        test_sessions = self.test_df['session_id'].unique()
        total_sessions = len(test_sessions)
        
        print(f"Generating predictions for {total_sessions} test sessions...")
        
        predictions = {}
        batch_size = 100  # Process in batches for progress display
        
        for i in range(0, total_sessions, batch_size):
            batch_sessions = test_sessions[i:i+batch_size]
            print(f"Processing sessions {i+1}-{min(i+batch_size, total_sessions)} of {total_sessions}...")
            
            for session_id in batch_sessions:
                try:
                    recommendations = self.recommend(session_id, n=5)
                    predictions[str(session_id)] = recommendations
                except Exception as e:
                    print(f"Error generating recommendations for session {session_id}: {str(e)}")
                    # Fallback to popular products
                    if 'overall' in self.features['popular_products']:
                        predictions[str(session_id)] = self.features['popular_products']['overall'][:5]
                    else:
                        # Last resort fallback
                        predictions[str(session_id)] = list(self.products_df['partnumber'].head(5))
        
        # Save predictions to file
        import json
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"Predictions saved to {output_path}")
        return predictions
    
    def validate(self, n_samples=100):
        """Validate recommender on a sample of train data"""
        if self.train_df is None:
            raise ValueError("Train data must be loaded")
        
        print(f"Validating recommender with {n_samples} sessions...")
        
        # Find sessions with cart additions
        cart_sessions = self.train_df[self.train_df['add_to_cart'] == 1]['session_id'].unique()
        
        if len(cart_sessions) == 0:
            print("No sessions with cart additions found for validation.")
            return pd.DataFrame()
        
        # Select random sample
        np.random.seed(42)  # For reproducibility
        selected_sessions = np.random.choice(cart_sessions, min(n_samples, len(cart_sessions)), replace=False)
        
        results = []
        for session_id in selected_sessions:
            # Get session data
            session_data = self.train_df[self.train_df['session_id'] == session_id].sort_values('timestamp_local')
            
            # Find last product added to cart
            cart_items = session_data[session_data['add_to_cart'] == 1]
            if cart_items.empty:
                continue
            
            last_cart_idx = cart_items.index[-1]
            history = session_data.loc[:last_cart_idx-1] if last_cart_idx > 0 else pd.DataFrame()
            target_product = session_data.loc[last_cart_idx, 'partnumber']
            
            # Use history as test data
            temp_test = self.test_df
            self.test_df = history
            
            # Generate recommendations
            recommendations = self.recommend(session_id, n=5)
            
            # Restore test data
            self.test_df = temp_test
            
            # Evaluate if target is in recommendations
            is_hit = target_product in recommendations
            hit_position = recommendations.index(target_product) + 1 if is_hit else 0
            
            results.append({
                'session_id': session_id,
                'target_product': target_product,
                'recommendations': recommendations,
                'is_hit': is_hit,
                'hit_position': hit_position
            })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, validation_results):
        """Calculate evaluation metrics"""
        if validation_results.empty:
            return {"error": "No validation results"}
        
        # Basic metrics
        metrics = {
            'num_sessions': len(validation_results),
            'hit_rate': validation_results['is_hit'].mean(),
        }
        
        # Calculate NDCG@5
        ndcg_scores = []
        for _, row in validation_results.iterrows():
            if row['is_hit']:
                position = row['hit_position']
                dcg = 1 / np.log2(position + 1)
                idcg = 1  # Ideal DCG (relevant item in first position)
                ndcg = dcg / idcg
            else:
                ndcg = 0
            
            ndcg_scores.append(ndcg)
        
        metrics['ndcg@5'] = np.mean(ndcg_scores) if ndcg_scores else 0
        return metrics