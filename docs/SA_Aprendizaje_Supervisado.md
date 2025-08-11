# Aprendizaje Supervisado - Documentación

## Justificación de Algoritmos

### Modelo de Clasificación - Random Forest
**¿Por qué Random Forest?**
- Maneja bien datos mixtos (numéricos y categóricos)
- Resistente al overfitting
- Proporciona importancia de características
- No requiere normalización estricta
- Robusto ante outliers

### Modelo de Regresión - Regresión Logística
**¿Por qué Regresión Logística?**
- Perfecta para problemas de clasificación binaria
- Interpretable (coeficientes indican importancia)
- Rápida de entrenar
- Probabilidades calibradas
- Base sólida para comparaciones

## Criterios de Análisis

### Criterio 1: Precisión y Recall
- **Heart Disease:** Precisión 85%+, Recall alto para detectar enfermedades
- **Income Prediction:** Precisión balanceada entre ambas clases

### Criterio 2: Interpretabilidad
- **Random Forest:** Feature importance muestra qué variables son más predictivas
- **Regresión Logística:** Coeficientes indican dirección e intensidad del impacto

## Archivos Generados
- `models/heart_disease_model.pkl` - Modelo de clasificación
- `models/income_regression_model.pkl` - Modelo de regresión