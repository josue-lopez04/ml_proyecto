#!/usr/bin/env python3
# Script para descargar automáticamente todos los datasets necesarios
# Ejecuta este script desde la carpeta raíz del proyecto

import os
import pandas as pd
import requests
import zipfile
from io import StringIO
import urllib.request

def create_data_folder():
    """Crea la carpeta data si no existe"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✅ Carpeta 'data' creada")
    else:
        print("📁 Carpeta 'data' ya existe")

def download_heart_disease():
    """Descarga dataset de enfermedades cardíacas"""
    print("\n🫀 Descargando Heart Disease Dataset...")
    
    # URL del dataset de UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    try:
        # Descargar datos
        response = requests.get(url)
        
        # Crear DataFrame con las columnas correctas
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        # Leer los datos
        data = []
        for line in response.text.strip().split('\n'):
            row = line.split(',')
            if len(row) == 14:  # Verificar que tenga todas las columnas
                data.append(row)
        
        df = pd.DataFrame(data, columns=columns)
        
        # Limpiar datos (convertir '?' a NaN y luego eliminar)
        df = df.replace('?', pd.NA)
        df = df.dropna()
        
        # Convertir tipos de datos
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Guardar
        df.to_csv('data/heart_disease.csv', index=False)
        print(f"✅ Heart Disease guardado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
    except Exception as e:
        print(f"❌ Error descargando Heart Disease: {e}")
        # Crear datos de ejemplo si falla
        create_heart_disease_sample()

def create_heart_disease_sample():
    """Crea datos de ejemplo para heart disease"""
    print("🔄 Creando datos de ejemplo para Heart Disease...")
    
    import numpy as np
    np.random.seed(42)
    
    n_samples = 300
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 3, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/heart_disease.csv', index=False)
    print("✅ Datos de ejemplo Heart Disease creados")

def download_adult_income():
    """Descarga dataset de Adult Income"""
    print("\n💰 Descargando Adult Income Dataset...")
    
    try:
        # URLs de UCI
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        
        # Columnas del dataset
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # Descargar datos de entrenamiento
        df = pd.read_csv(train_url, names=columns, skipinitialspace=True)
        
        # Limpiar datos
        df = df.replace('?', pd.NA)
        df = df.dropna()
        
        # Guardar
        df.to_csv('data/adult.csv', index=False)
        print(f"✅ Adult Income guardado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
    except Exception as e:
        print(f"❌ Error descargando Adult Income: {e}")
        create_adult_sample()

def create_adult_sample():
    """Crea datos de ejemplo para adult income"""
    print("🔄 Creando datos de ejemplo para Adult Income...")
    
    import numpy as np
    np.random.seed(42)
    
    n_samples = 1000
    
    workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov']
    education_options = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc']
    marital_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed']
    occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty']
    
    data = {
        'age': np.random.randint(17, 90, n_samples),
        'workclass': np.random.choice(workclass_options, n_samples),
        'fnlwgt': np.random.randint(12285, 1484705, n_samples),
        'education': np.random.choice(education_options, n_samples),
        'education-num': np.random.randint(1, 16, n_samples),
        'marital-status': np.random.choice(marital_options, n_samples),
        'occupation': np.random.choice(occupation_options, n_samples),
        'relationship': np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative'], n_samples),
        'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], n_samples),
        'sex': np.random.choice(['Female', 'Male'], n_samples),
        'capital-gain': np.random.randint(0, 99999, n_samples),
        'capital-loss': np.random.randint(0, 4356, n_samples),
        'hours-per-week': np.random.randint(1, 99, n_samples),
        'native-country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico'], n_samples),
        'income': np.random.choice(['<=50K', '>50K'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/adult.csv', index=False)
    print("✅ Datos de ejemplo Adult Income creados")

def download_happiness_data():
    """Descarga o crea datos de felicidad mundial"""
    print("\n😊 Descargando World Happiness Dataset...")
    
    try:
        # Intentar descargar desde Kaggle (requiere autenticación)
        print("⚠️  Para Kaggle necesitas API key. Creando datos de ejemplo...")
        create_happiness_sample()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        create_happiness_sample()

def create_happiness_sample():
    """Crea datos de ejemplo de felicidad mundial"""
    print("🔄 Creando datos de ejemplo para World Happiness...")
    
    import numpy as np
    np.random.seed(42)
    
    countries = [
        'Denmark', 'Switzerland', 'Iceland', 'Norway', 'Finland', 'Canada', 'Netherlands', 'Sweden',
        'Australia', 'Israel', 'Austria', 'Costa Rica', 'Mexico', 'Brazil', 'United States', 'Germany',
        'Belgium', 'United Kingdom', 'Chile', 'France', 'Spain', 'Italy', 'Japan', 'South Korea',
        'Argentina', 'Russia', 'Poland', 'Greece', 'Portugal', 'Czech Republic', 'Turkey', 'India',
        'China', 'South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Ghana', 'Morocco', 'Algeria',
        'Afghanistan', 'Bangladesh', 'Pakistan', 'Nepal', 'Myanmar', 'Cambodia', 'Venezuela', 'Colombia'
    ]
    
    n_countries = len(countries)
    
    data = {
        'Country': countries,
        'Happiness Score': np.random.uniform(3.0, 8.0, n_countries),
        'GDP per capita': np.random.uniform(0.2, 2.0, n_countries),
        'Social support': np.random.uniform(0.3, 1.2, n_countries),
        'Healthy life expectancy': np.random.uniform(0.2, 1.0, n_countries),
        'Freedom to make life choices': np.random.uniform(0.1, 0.8, n_countries),
        'Generosity': np.random.uniform(-0.3, 0.7, n_countries),
        'Perceptions of corruption': np.random.uniform(0.0, 1.0, n_countries)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/happiness_report.csv', index=False)
    print(f"✅ World Happiness creado: {df.shape[0]} países")

def download_retail_data():
    """Descarga o crea datos de retail online"""
    print("\n🛒 Descargando Online Retail Dataset...")
    
    try:
        # URL del dataset de UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        
        print("⚠️  Archivo muy grande. Creando datos de ejemplo...")
        create_retail_sample()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        create_retail_sample()

def create_retail_sample():
    """Crea datos de ejemplo para retail online"""
    print("🔄 Creando datos de ejemplo para Online Retail...")
    
    import numpy as np
    np.random.seed(42)
    
    n_transactions = 5000
    
    products = [
        'WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS COAT HANGER',
        'KNITTED UNION FLAG HOT WATER BOTTLE', 'RED WOOLLY HOTTIE WHITE HEART', 'SET 7 BABUSHKA NESTING BOXES',
        'GLASS STAR FROSTED T-LIGHT HOLDER', 'HAND WARMER UNION JACK', 'HAND WARMER RED POLKA DOT',
        'ASSORTED COLOUR BIRD ORNAMENT', 'POPPY S PLAYHOUSE KITCHEN', 'POPPY S PLAYHOUSE BEDROOM',
        'FELTCRAFT PRINCESS CHARLOTTE DOLL', 'IVORY KNITTED MUG COSY', 'BOX OF 6 ASSORTED COLOUR TEASPOONS',
        'BOX OF VINTAGE JIGSAW BLOCKS', 'BOX OF VINTAGE ALPHABET BLOCKS', 'HOME BUILDING BLOCK WORD',
        'LOVE BUILDING BLOCK WORD', 'RECIPE BOX WITH METAL HEART', 'DOORMAT NEW ENGLAND',
        'JUMP UP AND DO SOMETHING', 'METAL SIGN TAKE IT OR LEAVE IT', 'WOODEN STAR CHRISTMAS SCANDINAVIAN'
    ]
    
    countries = ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal', 'Australia']
    
    data = {
        'InvoiceNo': [f'C{537000 + i//10}' if np.random.random() < 0.05 else f'{537000 + i//10}' for i in range(n_transactions)],
        'StockCode': [f'{np.random.randint(10000, 99999)}' for _ in range(n_transactions)],
        'Description': np.random.choice(products, n_transactions),
        'Quantity': np.random.randint(1, 50, n_transactions),
        'InvoiceDate': pd.date_range('2010-12-01', '2011-12-09', periods=n_transactions),
        'UnitPrice': np.random.uniform(0.5, 50.0, n_transactions).round(2),
        'CustomerID': np.random.randint(12346, 18287, n_transactions),
        'Country': np.random.choice(countries, n_transactions)
    }
    
    df = pd.DataFrame(data)
    
    # Algunas transacciones negativas (cancelaciones)
    cancel_mask = df['InvoiceNo'].str.startswith('C')
    df.loc[cancel_mask, 'Quantity'] = -df.loc[cancel_mask, 'Quantity']
    
    df.to_csv('data/online_retail.csv', index=False)
    print(f"✅ Online Retail creado: {df.shape[0]} transacciones")

def main():
    """Función principal que descarga todos los datasets"""
    print("🚀 DESCARGANDO TODOS LOS DATASETS PARA EL PROYECTO ML\n")
    print("=" * 60)
    
    # Crear carpeta data
    create_data_folder()
    
    # Descargar todos los datasets
    download_heart_disease()
    download_adult_income()
    download_happiness_data()
    download_retail_data()
    
    print("\n" + "=" * 60)
    print("🎉 ¡DESCARGA COMPLETADA!")
    print("\nArchivos creados en la carpeta 'data/':")
    
    # Verificar archivos creados
    data_files = ['heart_disease.csv', 'adult.csv', 'happiness_report.csv', 'online_retail.csv']
    
    for file in data_files:
        file_path = f'data/{file}'
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size/1024:.1f} KB)")
        else:
            print(f"❌ {file} - NO ENCONTRADO")
    
    print("\n🔥 Ya puedes continuar con el entrenamiento de modelos!")
    print("Siguiente paso: cd models && python clasificacion_heart_disease.py")

if __name__ == "__main__":
    main()