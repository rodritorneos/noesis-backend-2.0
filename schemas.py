from pydantic import BaseModel
from typing import List, Optional, Any


# === USUARIO ===
class UsuarioRegistro(BaseModel):
    username: str
    email: str
    password: str


class UsuarioLogin(BaseModel):
    email: str
    password: str


class UsuarioResponse(BaseModel):
    username: str
    email: str
    password: str


class UsuarioInfo(BaseModel):
    username: str
    email: str


class UsuarioUpdateProfile(BaseModel):
    username: str


class UsuarioChangePassword(BaseModel):
    new_password: str


# === FAVORITO ===
class FavoritoRequest(BaseModel):
    clase_id: str
    nombre_clase: str
    imagen_path: str


class FavoritoResponse(BaseModel):
    clase_id: str
    nombre_clase: str
    imagen_path: str


class FavoritosUsuarioResponse(BaseModel):
    email: str
    favoritos: List[FavoritoResponse]
    total: int


# === VISITA ===
class VisitaRequest(BaseModel):
    clase_id: str


class VisitaResponse(BaseModel):
    clase_id: str
    count: int


class VisitasUsuarioResponse(BaseModel):
    email: str
    visitas: List[VisitaResponse]
    total_visitas: int


# === PUNTAJE ===
class PuntajeRequest(BaseModel):
    puntaje_obtenido: int
    puntaje_total: int
    nivel: str


class PuntajeResponse(BaseModel):
    email: str
    puntaje_obtenido: int
    puntaje_total: int
    nivel: str


class PuntajeUpdateResponse(BaseModel):
    message: str
    data: dict


# === RESPUESTAS GENERALES ===
class MessageResponse(BaseModel):
    message: str


class RegistroResponse(BaseModel):
    message: str
    email: str


class LoginResponse(BaseModel):
    message: str
    email: str


class FavoritoAddResponse(BaseModel):
    message: str
    favorito: FavoritoResponse


class HealthResponse(BaseModel):
    status: str
    database: str
    usuarios_registrados: int
    total_favoritos: int
    total_visitas: int
    total_puntajes: int
    database_ok: bool


class RootResponse(BaseModel):
    message: str
    version: str
    database: str
    endpoints: dict


# === ML MODEL ===
class ModelPredictRequest(BaseModel):
    puntaje_obtenido: int
    puntaje_total: int


class ModelPredictResponse(BaseModel):
    nivel_predicho: str
    porcentaje: float
    puntaje_obtenido: int
    puntaje_total: int
    probabilidades: Optional[dict] = None
    confianza: Optional[float] = None


class ModelStatsResponse(BaseModel):
    modelo_cargado: bool
    tipo_modelo: Optional[str] = None
    estadisticas: Any


class ModelPredictUserRequest(BaseModel):
    pass


class ModelPredictUserResponse(BaseModel):
    email: str
    nivel_predicho: str
    nivel_actual_guardado: str
    porcentaje: float
    puntaje_obtenido: int
    puntaje_total: int
    probabilidades: Optional[dict] = None
    confianza: Optional[float] = None
    es_prediccion_mejor: bool


class ModelUserStatsResponse(BaseModel):
    email: str
    puntaje_actual: dict
    prediccion_actual: dict
    estadisticas_modelo: dict
    recomendacion: str


# === DOCENTE ===
class DocenteRegistro(BaseModel):
    username: str
    email: str
    password: str
    institucion: str = "CCJCALLAO"


class DocenteLogin(BaseModel):
    email: str
    password: str


class DocenteResponse(BaseModel):
    username: str
    email: str
    password: str
    institucion: str


class DocenteInfo(BaseModel):
    username: str
    email: str
    institucion: str


# === ESTADÍSTICAS DE DOCENTE ===
class EstudianteEstadistica(BaseModel):
    username: str
    email: str
    puntaje_obtenido: int
    puntaje_total: int
    nivel: str
    fecha_ultima_actividad: str
    lecciones_completadas: int


class EstadisticasGenerales(BaseModel):
    promedio_general: float
    total_estudiantes: int
    estudiantes_activos: int
    distribucion_niveles: dict


class ProgresoEstudiantes(BaseModel):
    estudiantes: List[EstudianteEstadistica]
    total: int


class DashboardDocenteResponse(BaseModel):
    estadisticas_generales: EstadisticasGenerales
    progreso_estudiantes: ProgresoEstudiantes


# === PRONUNCIACIÓN (Nuevo) ===
class PronunciacionRequest(BaseModel):
    email: str
    tema: str
    file_path: str  # Ruta del archivo temporal (.wav)


class PronunciacionResponse(BaseModel):
    email: str
    tema: str
    frase_ref: str
    transcripcion: str
    similitud: int
    porcentaje: int
    feedback: str
    timestamp: str


class PronunciacionHistorialResponse(BaseModel):
    email: str
    historial: List[PronunciacionResponse]
