# Este script entrena un modelo para predecir enfermedades cardíacas
# Uso Random Forest porque maneja bien datos mixtos y da buena interpretabilidad

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class HeartDiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,    # 100 árboles para estabilidad
            random_state=42,     # Para reproducibilidad
            max_depth=10         # Evitar overfitting
        )
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, filepath):
        """Carga y prepara los datos para entrenamiento"""
        df = pd.read_csv(filepath)
        
        # Separar características y variable objetivo
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X, y
    
    def train(self, X, y):
        """Entrena el modelo"""
        # Dividir datos (80% entrenamiento, 20% prueba)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar características numéricas
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Hacer predicciones
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluar modelo
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Precisión del modelo: {accuracy:.3f}")
        print("Reporte de clasificación:")
        print(report)
        
        return X_test, y_test, y_pred
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Modelo guardado en: {filepath}")

# Uso del modelo
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    # Cargar datos
    X, y = predictor.load_and_prepare_data('../data/heart_disease.csv')
    
    # Entrenar modelo
    X_test, y_test, y_pred = predictor.train(X, y)
    
    # Guardar modelo
    predictor.save_model('../models/heart_disease_model.pkl')