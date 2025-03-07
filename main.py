"""
Main script to run the recommender system
"""
import os
import pandas as pd
import pickle
import json
import argparse
from src.data.processor import FeatureProcessor
from src.models.recommender import HybridRecommender


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='E-Commerce Recommender System')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train the model, do not generate predictions')
    parser.add_argument('--predict-only', action='store_true',
                        help='Only generate predictions, do not train the model')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation on a sample of train data')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    return parser.parse_args()


def main():
    """Main function to run the recommender system"""
    # Parse command line arguments
    args = parse_args()
    
    # Set data paths
    DATA_PATH = args.data_dir
    OUTPUT_PATH = args.output_dir
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(DATA_PATH, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "processed"), exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Set file paths
    train_path = os.path.join(DATA_PATH, "raw", "train.csv")
    products_path = os.path.join(DATA_PATH, "raw", "products.pkl")
    test_path = os.path.join(DATA_PATH, "raw", "test.csv")
    
    # Create feature processor
    feature_processor = FeatureProcessor()
    
    # Create hybrid recommender
    print("Initializing hybrid recommender system...")
    recommender = HybridRecommender(DATA_PATH, OUTPUT_PATH)
    
    # Load data
    recommender.load_data(train_path, products_path, test_path)
    
    # Process features
    print("Processing features...")
    recommender.process_features(feature_processor)
    
    # Train or load models
    model_path = os.path.join(DATA_PATH, "models", "collaborative_model.pt")
    if os.path.exists(model_path) and (args.predict_only or not args.train_only):
        print("Loading pre-trained models...")
        recommender.load_trained_models()
    elif not args.predict_only:
        print("Training models...")
        recommender.train_models(train_cf=True)
    
    # Validate on a sample if requested
    if args.validate:
        print("Validating model...")
        validation_results = recommender.validate(n_samples=200)
        
        if not validation_results.empty:
            metrics = recommender.calculate_metrics(validation_results)
            print("Validation metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    # Generate predictions for test if not in train-only mode
    if not args.train_only:
        print("Generating predictions for test set...")
        predictions_path = os.path.join(OUTPUT_PATH, "predictions_3.json")
        recommender.generate_all_predictions(predictions_path)
    
    print("Recommender system process completed successfully!")
    
    # Also generate predictions_1.json for Task 1
    task1_path = os.path.join(OUTPUT_PATH, "predictions_1.json")
    if not os.path.exists(task1_path):
        print("Note: predictions_1.json for Task 1 not found.")
        print("You should implement the data analysis queries and save the results to predictions_1.json")


def answer_task1_questions():
    """
    Helper function to answer Task 1 questions and generate predictions_1.json
    This should be implemented separately for Task 1
    """
    # Implementation for Task 1 queries would go here
    # Example structure:
    # results = {
    #     'q1': ...,
    #     'q2': ...,
    #     'q3': ...,
    #     'q4': ...,
    #     'q5': ...,
    #     'q6': ...,
    #     'q7': ...,
    # }
    
    # Save results to predictions_1.json
    # with open('predictions/predictions_1.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    pass


if __name__ == "__main__":
    # Write Python as language choice
    with open('language.txt', 'w') as f:
        f.write('Python')
    
    # Run the main function
    main()