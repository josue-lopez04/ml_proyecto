# Agrupación de países por nivel de felicidad
# Uso K-Means porque queremos encontrar grupos naturales de países similares

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class HappinessClusterer:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10              # Múltiples inicializaciones para estabilidad
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # Para visualización
        
    def load_and_prepare_data(self, filepath):
        """Carga y prepara datos de felicidad mundial"""
        df = pd.read_csv(filepath)
        
        # Seleccionar características numéricas relevantes
        feature_columns = [
            'GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
        ]
        
        # Limpiar datos
        df_clean = df[['Country'] + feature_columns].dropna()
        
        return df_clean
    
    def fit_clusters(self, df):
        """Aplica K-Means clustering"""
        # Preparar datos
        features = df.drop('Country', axis=1)
        
        # Normalizar características
        features_scaled = self.scaler.fit_transform(features)
        
        # Aplicar clustering
        clusters = self.model.fit_predict(features_scaled)
        
        # Añadir clusters al dataframe
        df_result = df.copy()
        df_result['Cluster'] = clusters
        
        # PCA para visualización
        pca_features = self.pca.fit_transform(features_scaled)
        df_result['PC1'] = pca_features[:, 0]
        df_result['PC2'] = pca_features[:, 1]
        
        return df_result
    
    def analyze_clusters(self, df_clustered):
        """Analiza las características de cada cluster"""
        print("=== ANÁLISIS DE CLUSTERS ===\n")
        
        for i in range(self.n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == i]
            print(f"CLUSTER {i} ({len(cluster_data)} países):")
            print("Países representativos:")
            print(cluster_data['Country'].head().tolist())
            
            # Estadísticas del cluster
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['Cluster', 'PC1', 'PC2']]
            
            print("\nCaracterísticas promedio:")
            for col in numeric_cols:
                avg = cluster_data[col].mean()
                print(f"  {col}: {avg:.3f}")
            print("-" * 50)
    
    def visualize_clusters(self, df_clustered):
        """Crea visualizaciones de los clusters"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot con PCA
        plt.subplot(2, 2, 1)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(self.n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == i]
            plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                       c=colors[i], label=f'Cluster {i}', alpha=0.7)
        
        plt.xlabel('Primera Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.title('Clusters de Países por Felicidad (PCA)')
        plt.legend()
        
        # Distribución por cluster
        plt.subplot(2, 2, 2)
        df_clustered['Cluster'].value_counts().sort_index().plot(kind='bar')
        plt.title('Número de Países por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Cantidad de Países')
        
        # Heatmap de características promedio
        plt.subplot(2, 1, 2)
        numeric_cols = ['GDP per capita', 'Social support', 'Healthy life expectancy',
                       'Freedom to make life choices', 'Generosity']
        
        cluster_means = []
        for i in range(self.n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == i]
            means = [cluster_data[col].mean() for col in numeric_cols]
            cluster_means.append(means)
        
        cluster_df = pd.DataFrame(cluster_means, 
                                 columns=numeric_cols,
                                 index=[f'Cluster {i}' for i in range(self.n_clusters)])
        
        sns.heatmap(cluster_df, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Características Promedio por Cluster')
        
        plt.tight_layout()
        plt.show()
        
        return df_clustered
    
    def save_model(self, filepath):
        """Guarda el modelo de clustering"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'n_clusters': self.n_clusters
            }, f)
        print(f"Modelo de clustering guardado en: {filepath}")

# Ejecutar clustering
if __name__ == "__main__":
    clusterer = HappinessClusterer(n_clusters=4)
    
    # Cargar datos
    df = clusterer.load_and_prepare_data('../data/happiness_report.csv')
    
    # Aplicar clustering
    df_clustered = clusterer.fit_clusters(df)
    
    # Analizar resultados
    clusterer.analyze_clusters(df_clustered)
    
    # Visualizar
    df_final = clusterer.visualize_clusters(df_clustered)
    
    # Guardar resultados
    df_final.to_csv('../data/happiness_clustered.csv', index=False)
    clusterer.save_model('../models/happiness_clustering_model.pkl')