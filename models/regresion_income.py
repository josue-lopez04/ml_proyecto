# Modelo para predecir si alguien gana más de $50K
# Uso Regresión Logística por su interpretabilidad en problemas binarios

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

class IncomePredictor:
    def __init__(self):
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000        # Aumentar iteraciones para convergencia
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_data(self, df):
        """Prepara los datos categóricos y numéricos"""
        df_processed = df.copy()
        
        # Codificar variables categóricas
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'income':  # No codificar la variable objetivo aún
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
        
        return df_processed
    
    def train(self, filepath):
        """Entrena el modelo de regresión"""
        # Cargar datos
        df = pd.read_csv(filepath)
        
        # Preprocesar datos
        df_processed = self.prepare_data(df)
        
        # Preparar X e y
        X = df_processed.drop('income', axis=1)
        y = (df_processed['income'] == '>50K').astype(int)  # Convertir a binario
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Precisión del modelo de regresión: {accuracy:.3f}")
        print("Reporte detallado:")
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test, y_pred
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.label_encoders
            }, f)
        print(f"Modelo de regresión guardado en: {filepath}")

# Entrenar modelo
if __name__ == "__main__":
    predictor = IncomePredictor()
    X_test, y_test, y_pred = predictor.train('../data/adult.csv')
    predictor.save_model('../models/income_regression_model.pkl')