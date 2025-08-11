# Aprendizaje No Supervisado - Documentación

## Justificación de Algoritmos

### Algoritmo de Agrupación - K-Means
**¿Por qué K-Means?**
- Eficiente para datasets medianos
- Interpretable: centroides representan características típicas
- Funciona bien con datos numéricos normalizados
- Permite determinar patrones de felicidad por regiones

### Algoritmo de Asociación - Apriori
**¿Por qué Apriori?**
- Estándar de la industria para Market Basket Analysis
- Genera reglas interpretables (Si X entonces Y)
- Permite ajustar soporte y confianza según necesidades del negocio
- Ideal para recomendaciones de productos

## Criterios de Análisis

### Criterio 1: Cohesión de Clusters
- **Silhouette Score:** Mide qué tan bien separados están los clusters
- **Inertia:** Suma de distancias al centroide (menor es mejor)
- **Interpretabilidad:** Los clusters deben tener sentido geográfico/económico

### Criterio 2: Calidad de Reglas de Asociación
- **Lift > 1:** La regla es mejor que el azar
- **Confidence > 0.3:** Al menos 30% de certeza
- **Support > 0.02:** Al menos 2% de transacciones contienen el patrón

## Archivos Generados
- `models/happiness_clustering_model.pkl` - Modelo de agrupación
- `models/retail_association_model.pkl` - Modelo de asociación
- `data/happiness_clustered.csv` - Países con sus clusters asignados
- `data/association_rules.csv` - Reglas de asociación encontradas