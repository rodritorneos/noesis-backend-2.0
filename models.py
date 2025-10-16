from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base


# === MODELO USUARIO ===
class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

    favoritos = relationship("Favorito", back_populates="usuario", cascade="all, delete-orphan")
    visitas = relationship("Visita", back_populates="usuario", cascade="all, delete-orphan")
    puntajes = relationship("Puntaje", back_populates="usuario", cascade="all, delete-orphan")

    # 👇 Nueva relación con Pronunciación
    pronunciaciones = relationship("Pronunciacion", back_populates="usuario", cascade="all, delete-orphan")


# === MODELO FAVORITO ===
class Favorito(Base):
    __tablename__ = "favoritos"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"))
    clase_id = Column(String, nullable=False)
    nombre_clase = Column(String, nullable=False)
    imagen_path = Column(String, nullable=False)

    usuario = relationship("Usuario", back_populates="favoritos")


# === MODELO VISITA ===
class Visita(Base):
    __tablename__ = "visitas"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"))
    clase_id = Column(String, nullable=False)
    count = Column(Integer, default=0)

    usuario = relationship("Usuario", back_populates="visitas")


# === MODELO PUNTAJE ===
class Puntaje(Base):
    __tablename__ = "puntajes"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"))
    puntaje_obtenido = Column(Integer, default=0)
    puntaje_total = Column(Integer, default=0)
    nivel = Column(String, default="A1")

    usuario = relationship("Usuario", back_populates="puntajes")


# === MODELO DOCENTE ===
class Docente(Base):
    __tablename__ = "docentes"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    institucion = Column(String, nullable=False, default="CCJCALLAO")


# === MODELO DE PRONUNCIACIÓN (Nuevo) ===
class Pronunciacion(Base):
    __tablename__ = "pronunciaciones"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    tema = Column(String, nullable=False)
    frase_ref = Column(String, nullable=False)
    transcripcion = Column(String, nullable=False)
    similitud = Column(Integer, default=0)       # Valor entre 0 y 100
    porcentaje = Column(Integer, default=0)
    feedback = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)

    usuario = relationship("Usuario", back_populates="pronunciaciones")
