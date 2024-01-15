from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def get_evaluation(y_test, probs, imprime = False, treshold = 0.5):
    """
    Evalúa el rendimiento de un modelo de clasificación binaria mediante diversas métricas.

    Parámetros:
    - y_test (array): Etiquetas binarias reales.
    - probs (array): Probabilidades de pertenecer a la clase 1.

    Retorna:
    - brier_score: Puntaje Brier, una medida de la precisión de las probabilidades predichas.

    Imprime:
    - Classification Report: Un informe detallado de métricas de clasificación.
    - Brier Score: El puntaje Brier calculado.
    - Confusion Matrix: Una matriz de confusión visualizada como un mapa de calor.

    Nota:
    - Se considera que las instancias con probabilidades mayores o iguales a 0.5 pertenecen a la clase 1.
    Requiere:
    - classification_report: Función para generar un informe de clasificación (de scikit-learn.metrics).
    - brier_score_loss: Función para calcular el puntaje Brier (de scikit-learn.metrics).
    - confusion_matrix: Función para calcular la matriz de confusión (de scikit-learn.metrics).
    - sns: Seaborn, una biblioteca de visualización estadística.
    - plt: Matplotlib, una biblioteca de visualización.

    """
    # Asignamos etiquetas
    y_pred = (probs >= treshold).astype(int)
    # Reporte de clasificación
    classification_rep = classification_report(y_test, y_pred)    
    # Calcular el Brier Score
    brier_score = brier_score_loss(y_test, probs)    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    if imprime:
        print('\nClassification Report:\n', classification_rep)
        print(f'Brier Score: {brier_score:.6f}')
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)        
        # Añadir etiquetas y título
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Mostrar el mapa de calor
        plt.show()
    
    return brier_score
