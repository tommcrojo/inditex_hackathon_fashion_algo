import os
import sys
import pandas as pd
import numpy as np
import pickle
import json

# Importar las clases y funciones necesarias
from src.models.recommender import HybridRecommender
from src.data.session_metrics import get_session_metrics

def main():
    """Main function to run the recommender system"""
    # Set paths
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")  
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "predictions")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(DATA_PATH, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "processed"), exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    train_path = os.path.join(DATA_PATH, "raw", "train.csv")
    products_path = os.path.join(DATA_PATH, "raw", "products.pkl")
    test_path = os.path.join(DATA_PATH, "raw", "test.csv")
    
    # Verificar que los archivos existan
    for file_path in [train_path, products_path, test_path]:
        if not os.path.exists(file_path):
            print(f"Error: Archivo no encontrado: {file_path}")
            return
    
    # Create hybrid recommender
    print("Inicializando sistema de recomendación híbrido...")
    recommender = HybridRecommender(DATA_PATH, OUTPUT_PATH)
    
    # Load data
    recommender.load_data(train_path, products_path, test_path)
    
    # Process features
    print("Procesando características...")
    recommender.process_features()
    
    # Train or load models
    model_path = os.path.join(DATA_PATH, "models", "collaborative_model.pt")
    if os.path.exists(model_path):
        print("Cargando modelos pre-entrenados...")
        recommender.load_trained_models()
    else:
        print("Entrenando modelos (puede tomar tiempo)...")
        recommender.train_models()
    
    # Validate on a sample
    print("Validando modelo...")
    validation_results = recommender.validate(n_samples=100)
    
    if not validation_results.empty:
        metrics = recommender.calculate_metrics(validation_results)
        print("Métricas de validación:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Generate predictions for test
    print("Generando predicciones para el conjunto de prueba...")
    predictions_path = os.path.join(OUTPUT_PATH, "predictions_3.json")
    recommender.generate_all_predictions(predictions_path)
    
    print("¡Proceso completado con éxito!")
    print(f"Predicciones guardadas en: {predictions_path}")

if __name__ == "__main__":
    main()