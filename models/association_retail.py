# Análisis de asociación para encontrar productos que se compran juntos
# Uso algoritmo Apriori para encontrar reglas de asociación útiles

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import pickle

class MarketBasketAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.3):
        self.min_support = min_support      # Mínimo 1% de las transacciones
        self.min_confidence = min_confidence  # 30% de confianza mínima
        self.frequent_itemsets = None
        self.rules = None
        
    def load_and_prepare_data(self, filepath):
        """Carga y prepara datos de retail para análisis de asociación"""
        df = pd.read_csv(filepath, encoding='latin1')  # A veces necesario para archivos UCI
        
        # Limpiar datos
        df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description'])
        df = df[df['Quantity'] > 0]  # Solo compras positivas
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]  # Excluir cancelaciones
        
        # Crear transacciones (agrupar por factura)
        transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
        
        # Limpiar nombres de productos (quitar espacios extra, convertir a minúsculas)
        transactions_clean = []
        for transaction in transactions:
            clean_transaction = [item.strip().lower() for item in transaction if isinstance(item, str)]
            transactions_clean.append(clean_transaction)
        
        return transactions_clean
    
    def create_basket_matrix(self, transactions):
        """Convierte transacciones a matriz binaria"""
        # Usar TransactionEncoder de mlxtend
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_basket
    
    def find_frequent_itemsets(self, df_basket):
        """Encuentra conjuntos de ítems frecuentes"""
        # Aplicar algoritmo Apriori
        frequent_itemsets = apriori(df_basket, 
                                   min_support=self.min_support, 
                                   use_colnames=True)
        
        # Ordenar por soporte descendente
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        
        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets
    
    def generate_association_rules(self):
        """Genera reglas de asociación desde los itemsets frecuentes"""
        if self.frequent_itemsets is None:
            raise ValueError("Primero debes encontrar itemsets frecuentes")
        
        # Generar reglas con diferentes métricas
        rules = association_rules(self.frequent_itemsets, 
                                 metric="confidence", 
                                 min_threshold=self.min_confidence)
        
        # Ordenar por lift (relevancia) descendente
        rules = rules.sort_values('lift', ascending=False)
        
        self.rules = rules
        return rules
    
    def analyze_results(self):
        """Analiza y muestra los resultados más interesantes"""
        print("=== ANÁLISIS DE MARKET BASKET ===\n")
        
        # Top 10 itemsets más frecuentes
        print("TOP 10 PRODUCTOS/COMBOS MÁS FRECUENTES:")
        top_items = self.frequent_itemsets.head(10)
        for idx, row in top_items.iterrows():
            items = ', '.join(list(row['itemsets']))
            print(f"  {items} (soporte: {row['support']:.3f})")
        
        print("\n" + "="*60)
        
        # Top 10 reglas más relevantes
        print("TOP 10 REGLAS MÁS RELEVANTES (por lift):")
        top_rules = self.rules.head(10)
        for idx, rule in top_rules.iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            print(f"  Si compra: {antecedent}")
            print(f"  También comprará: {consequent}")
            print(f"  Confianza: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
            print("-" * 40)
    
    def visualize_results(self):
        """Crea visualizaciones de los resultados"""
        plt.figure(figsize=(15, 10))
        
        # Distribución de soporte
        plt.subplot(2, 3, 1)
        self.frequent_itemsets['support'].hist(bins=20)
        plt.title('Distribución del Soporte')
        plt.xlabel('Soporte')
        plt.ylabel('Frecuencia')
        
        # Top 15 itemsets por soporte
        plt.subplot(2, 3, 2)
        top_15 = self.frequent_itemsets.head(15)
        item_names = [', '.join(list(itemset))[:30] + '...' if len(', '.join(list(itemset))) > 30 
                     else ', '.join(list(itemset)) for itemset in top_15['itemsets']]
        
        plt.barh(range(len(item_names)), top_15['support'])
        plt.yticks(range(len(item_names)), item_names, fontsize=8)
        plt.title('Top 15 Itemsets por Soporte')
        plt.xlabel('Soporte')
        
        # Scatter plot de reglas (confianza vs lift)
        plt.subplot(2, 3, 3)
        plt.scatter(self.rules['confidence'], self.rules['lift'], alpha=0.6)
        plt.xlabel('Confianza')
        plt.ylabel('Lift')
        plt.title('Confianza vs Lift de Reglas')
        
        # Distribución de lift
        plt.subplot(2, 3, 4)
        self.rules['lift'].hist(bins=20)
        plt.title('Distribución del Lift')
        plt.xlabel('Lift')
        plt.ylabel('Frecuencia')
        
        # Top reglas por lift
        plt.subplot(2, 3, 5)
        top_rules_lift = self.rules.head(10)
        rule_names = [f"{', '.join(list(rule['antecedents']))} → {', '.join(list(rule['consequents']))}"[:25] + '...' 
                     for idx, rule in top_rules_lift.iterrows()]
        
        plt.barh(range(len(rule_names)), top_rules_lift['lift'])
        plt.yticks(range(len(rule_names)), rule_names, fontsize=8)
        plt.title('Top 10 Reglas por Lift')
        plt.xlabel('Lift')
        
        # Heatmap de métricas
        plt.subplot(2, 3, 6)
        metrics_sample = self.rules.head(20)[['support', 'confidence', 'lift']]
        plt.imshow(metrics_sample.T, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Heatmap de Métricas (Top 20 Reglas)')
        plt.yticks([0, 1, 2], ['Support', 'Confidence', 'Lift'])
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Guarda los resultados del análisis"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'frequent_itemsets': self.frequent_itemsets,
                'rules': self.rules,
                'min_support': self.min_support,
                'min_confidence': self.min_confidence
            }, f)
        print(f"Modelo de asociación guardado en: {filepath}")

# Ejecutar análisis de asociación
if __name__ == "__main__":
    analyzer = MarketBasketAnalyzer(min_support=0.02, min_confidence=0.3)
    
    # Cargar y preparar datos
    transactions = analyzer.load_and_prepare_data('../data/online_retail.csv')
    print(f"Total de transacciones: {len(transactions)}")
    
    # Crear matriz de cestas
    df_basket = analyzer.create_basket_matrix(transactions)
    print(f"Productos únicos: {df_basket.shape[1]}")
    
    # Encontrar itemsets frecuentes
    frequent_items = analyzer.find_frequent_itemsets(df_basket)
    print(f"Itemsets frecuentes encontrados: {len(frequent_items)}")
    
    # Generar reglas de asociación
    rules = analyzer.generate_association_rules()
    print(f"Reglas de asociación generadas: {len(rules)}")
    
    # Analizar resultados
    analyzer.analyze_results()
    
    # Visualizar
    analyzer.visualize_results()
    
    # Guardar modelo
    analyzer.save_model('../models/retail_association_model.pkl')
    
    # Guardar resultados en CSV
    frequent_items.to_csv('../data/frequent_itemsets.csv', index=False)
    rules.to_csv('../data/association_rules.csv', index=False)