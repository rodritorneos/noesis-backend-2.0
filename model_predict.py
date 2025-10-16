import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class ModelPredictor:
    def __init__(self, csv_path: str = "dataset_noesis.csv"):
        self.model = None
        self.model_stats = None
        self.csv_path = csv_path
        self.feature_names = ["puntaje_obtenido", "puntaje_total", "relacion_puntaje", "clase_mas_recurrida_cod"]
        self.level_mapping = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
        self.reverse_mapping = {0: "Básico", 1: "Intermedio", 2: "Avanzado"}
        
        # Mapeo de clases de texto a código (basado en tu CSV)
        self.class_text_mapping = {
            "Verb to be": 0,
            "Present Simple": 1, 
            "The verb can": 2,
            "Future Perfect": 3
        }
        self.reverse_class_mapping = {0: "Verb to be", 1: "Present Simple", 2: "The verb can", 3: "Future Perfect"}
        
        self.train_model()
    
    def load_real_data(self) -> pd.DataFrame:
        """
        Cargar datos reales desde el CSV
        """
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"No se encontró el archivo: {self.csv_path}")
            
            print(f"📁 Cargando datos desde: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            print(f"✅ Datos cargados: {len(df)} registros")
            print(f"📊 Columnas encontradas: {list(df.columns)}")
            
            # Validar columnas requeridas
            required_columns = ["puntaje_obtenido", "puntaje_total", "relacion_puntaje", 
                              "clase_mas_recurrida_cod", "nivel"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Columnas faltantes en el CSV: {missing_columns}")
            
            # Validar niveles
            valid_levels = set(self.level_mapping.keys())
            invalid_levels = set(df["nivel"].unique()) - valid_levels
            if invalid_levels:
                print(f"⚠️ Niveles inválidos encontrados: {invalid_levels}")
                df = df[df["nivel"].isin(valid_levels)]
                print(f"📝 Registros válidos después del filtrado: {len(df)}")
            
            # Mostrar estadísticas del dataset
            print(f"\n📈 ESTADÍSTICAS DEL DATASET:")
            print(f"   • Total registros: {len(df)}")
            print(f"   • Distribución por nivel:")
            for nivel, count in df["nivel"].value_counts().items():
                print(f"     - {nivel}: {count} ({count/len(df)*100:.1f}%)")
            
            if "clase_mas_recurrida_txt" in df.columns:
                print(f"   • Clases más frecuentes:")
                for clase, count in df["clase_mas_recurrida_txt"].value_counts().head().items():
                    print(f"     - {clase}: {count}")
            
            print(f"   • Rango de puntajes: {df['puntaje_obtenido'].min()}-{df['puntaje_obtenido'].max()}")
            print(f"   • Rango de porcentajes: {df['relacion_puntaje'].min():.2f}-{df['relacion_puntaje'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error cargando datos reales: {e}")
            raise e
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generar datos sintéticos como fallback (manteniendo la estructura original)
        """
        np.random.seed(42)
        data = []
        
        for _ in range(n_samples):
            # Generar puntajes realistas
            puntaje_total = np.random.choice([10, 15, 20, 25, 30])
            
            # Generar nivel primero para asegurar coherencia
            nivel = np.random.choice(["Básico", "Intermedio", "Avanzado"], p=[0.4, 0.4, 0.2])
            
            # Generar puntaje_obtenido basado en el nivel
            if nivel == "Básico":
                puntaje_obtenido = np.random.randint(0, int(puntaje_total * 0.5))
            elif nivel == "Intermedio":
                puntaje_obtenido = np.random.randint(int(puntaje_total * 0.4), int(puntaje_total * 0.8))
            else:  # Avanzado
                puntaje_obtenido = np.random.randint(int(puntaje_total * 0.7), puntaje_total + 1)
            
            # Calcular relación de puntaje
            relacion_puntaje = puntaje_obtenido / puntaje_total if puntaje_total > 0 else 0
            
            # Generar clase más recurrida (codificada)
            clase_mas_recurrida_cod = np.random.randint(0, 4)
            
            data.append({
                "puntaje_obtenido": puntaje_obtenido,
                "puntaje_total": puntaje_total,
                "relacion_puntaje": relacion_puntaje,
                "clase_mas_recurrida_cod": clase_mas_recurrida_cod,
                "nivel": nivel
            })
        
        return pd.DataFrame(data)
    
    def train_model(self):
        """
        Entrenar el modelo con datos reales o sintéticos como fallback
        """
        try:
            # Intentar cargar datos reales primero
            try:
                df = self.load_real_data()
                data_source = "real_data"
                print("🎯 Usando datos reales para entrenamiento")
            except Exception as e:
                print(f"⚠️ No se pudieron cargar datos reales: {e}")
                print("🔄 Generando datos sintéticos como fallback...")
                df = self.generate_synthetic_data(2000)
                data_source = "synthetic_data"
            
            # Verificar que tenemos suficientes datos
            if len(df) < 10:
                print("⚠️ Datos insuficientes, complementando con datos sintéticos...")
                synthetic_df = self.generate_synthetic_data(1000)
                df = pd.concat([df, synthetic_df], ignore_index=True)
                data_source = "mixed_data"
            
            # Codificar la variable objetivo
            df["nivel_cod"] = df["nivel"].map(self.level_mapping)
            
            # Variables predictoras
            X = df[self.feature_names]
            y = df["nivel_cod"]
            
            # Verificar que no hay valores nulos
            if X.isnull().sum().sum() > 0:
                print("⚠️ Encontrados valores nulos, eliminando filas...")
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
            
            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entrenar el modelo
            print("🤖 Entrenando modelo Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train, y_train)
            
            # Evaluar el modelo
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Guardar estadísticas
            self.model_stats = {
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_samples_total": len(df),
                "data_source": data_source,
                "feature_importance": dict(zip(
                    self.feature_names, 
                    self.model.feature_importances_.round(4)
                )),
                "class_distribution": df["nivel"].value_counts().to_dict()
            }
            
            print(f"\n✅ MODELO ENTRENADO EXITOSAMENTE!")
            print(f"📊 Fuente de datos: {data_source}")
            print(f"📝 Total de datos: {len(df)}")
            print(f"🎯 Precisión en entrenamiento: {train_accuracy:.4f}")
            print(f"🎯 Precisión en prueba: {test_accuracy:.4f}")
            print(f"📈 Distribución de clases: {df['nivel'].value_counts().to_dict()}")
            print(f"🔍 Importancia de características:")
            for feature, importance in self.model_stats["feature_importance"].items():
                print(f"   • {feature}: {importance:.4f}")
            
        except Exception as e:
            print(f"❌ Error al entrenar el modelo: {e}")
            raise e
    
    def predict_level(self, score_obtained: int, total_score: int, 
                     most_frequent_class_code: int = 0) -> Dict[str, Any]:
        """
        Predecir el nivel de inglés basado en el puntaje
        
        Args:
            score_obtained: Puntaje obtenido por el estudiante
            total_score: Puntaje total posible
            most_frequent_class_code: Código de la clase más frecuente (0-3)
                0: Verb to be, 1: Present Simple, 2: The verb can, 3: Future Perfect
        """
        if not self.model:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Validar entradas
            if total_score <= 0:
                raise ValueError("El puntaje total debe ser mayor a 0")
            
            if score_obtained < 0 or score_obtained > total_score:
                raise ValueError("El puntaje obtenido debe estar entre 0 y el puntaje total")
            
            if most_frequent_class_code not in [0, 1, 2, 3]:
                most_frequent_class_code = 0  # Valor por defecto
            
            # Calcular relación de puntaje
            score_ratio = score_obtained / total_score
            
            # Preparar características para predicción
            features = np.array([[
                score_obtained,
                total_score,
                score_ratio,
                most_frequent_class_code
            ]])
            
            # Realizar predicción
            prediction_code = self.model.predict(features)[0]
            predicted_level = self.reverse_mapping[prediction_code]
            
            # Obtener probabilidades
            probabilities = self.model.predict_proba(features)[0]
            probability_dict = {
                self.reverse_mapping[i]: round(float(prob), 4) 
                for i, prob in enumerate(probabilities)
            }
            
            # Calcular porcentaje
            percentage = round(score_ratio * 100, 2)
            
            # Determinar confianza
            confidence = round(max(probabilities), 4)
            
            # Obtener nombre de la clase
            class_name = self.reverse_class_mapping.get(most_frequent_class_code, "Desconocido")
            
            return {
                "nivel_predicho": predicted_level,
                "codigo_nivel": int(prediction_code),
                "porcentaje": percentage,
                "puntaje_obtenido": score_obtained,
                "puntaje_total": total_score,
                "probabilidades": probability_dict,
                "confianza": confidence,
                "clase_mas_frecuente": most_frequent_class_code,
                "clase_mas_frecuente_nombre": class_name
            }
            
        except Exception as e:
            raise ValueError(f"Error en la predicción: {e}")
    
    def predict_batch(self, students_data: list) -> list:
        """
        Predecir niveles para múltiples estudiantes
        
        Args:
            students_data: Lista de diccionarios con datos de estudiantes
                         Cada diccionario debe tener: score_obtained, total_score, most_frequent_class_code
        """
        results = []
        for student in students_data:
            try:
                prediction = self.predict_level(
                    student.get('score_obtained'),
                    student.get('total_score'),
                    student.get('most_frequent_class_code', 0)
                )
                results.append(prediction)
            except Exception as e:
                results.append({"error": str(e), "student_data": student})
        
        return results
    
    def predict_from_class_name(self, score_obtained: int, total_score: int, 
                               class_name: str = "Verb to be") -> Dict[str, Any]:
        """
        Predecir usando el nombre de la clase en lugar del código
        
        Args:
            score_obtained: Puntaje obtenido
            total_score: Puntaje total
            class_name: Nombre de la clase ("Verb to be", "Present Simple", etc.)
        """
        class_code = self.class_text_mapping.get(class_name, 0)
        return self.predict_level(score_obtained, total_score, class_code)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del modelo"""
        if not self.model_stats:
            return {"error": "Modelo no entrenado"}
        
        return {
            **self.model_stats,
            "modelo_tipo": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "clases": list(self.reverse_mapping.values()),
            "caracteristicas": self.feature_names,
            "mapeo_clases": self.reverse_class_mapping
        }
    
    def get_level_distribution(self, predictions: list) -> Dict[str, int]:
        """
        Obtener distribución de niveles predichos
        """
        distribution = {"Básico": 0, "Intermedio": 0, "Avanzado": 0}
        
        for pred in predictions:
            if isinstance(pred, dict) and "nivel_predicho" in pred:
                level = pred["nivel_predicho"]
                if level in distribution:
                    distribution[level] += 1
        
        return distribution
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre el dataset cargado
        """
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                return {
                    "archivo": self.csv_path,
                    "total_registros": len(df),
                    "columnas": list(df.columns),
                    "distribucion_niveles": df["nivel"].value_counts().to_dict() if "nivel" in df.columns else {},
                    "clases_disponibles": df["clase_mas_recurrida_txt"].unique().tolist() if "clase_mas_recurrida_txt" in df.columns else [],
                    "rango_puntajes": {
                        "min": int(df["puntaje_obtenido"].min()) if "puntaje_obtenido" in df.columns else 0,
                        "max": int(df["puntaje_obtenido"].max()) if "puntaje_obtenido" in df.columns else 0
                    }
                }
            else:
                return {"error": f"Archivo {self.csv_path} no encontrado"}
        except Exception as e:
            return {"error": f"Error leyendo dataset: {e}"}

# Instancia global del predictor
print("🚀 Inicializando ModelPredictor con dataset real...")
predictor = ModelPredictor()
print("✅ ModelPredictor listo para usar!")