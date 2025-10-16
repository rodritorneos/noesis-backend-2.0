import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import noisereduce as nr
import requests
import json
from gtts import gTTS
from pydub import AudioSegment
from pydub.utils import which
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher
from datetime import datetime
import re


# === CONFIGURACIÓN GENERAL ===
REF_FILE = "voz_ref.wav"
USER_FILE = "voz_user.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELO = "gemma3:4b-it-qat"
URL_API = "http://localhost:11434/v1/chat/completions"


class PronunciationAnalyzer:
    def __init__(self):
        """Inicializa el modelo ASR y prepara las variables."""
        print("🧠 Cargando modelo Wav2Vec2 (ASR)...")
        self.processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        ).to(DEVICE)
        print("✅ Modelo de voz cargado correctamente.\n")

        self.temas = ["Verb to be", "Present Simple", "The verb can", "Future Perfect"]
        self.usadas = set()

    # === Generar oración simple A1–A2 ===
    def generar_oracion_tema(self, tema: str) -> str:
        """Genera una oración corta y simple para el tema seleccionado usando Ollama."""
        prompt = (
            f"Generate ONE very short and simple English sentence (A1–A2 level) for the topic '{tema}'. "
            f"Example: 'He is happy.' or 'I can run fast.' Avoid repetition or complex words."
        )

        response = requests.post(
            URL_API,
            json={
                "model": MODELO,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            },
        )

        data = response.json()
        frase = data["choices"][0]["message"]["content"].strip()

        frase = re.sub(r'[*"`]', "", frase).strip()
        frase = frase[0].upper() + frase[1:]
        if not frase.endswith("."):
            frase += "."

        if frase.lower() in self.usadas:
            return self.generar_oracion_tema(tema)

        self.usadas.add(frase.lower())
        return frase

    # === Generar TTS ===
    def generar_tts(self, frase, filename_wav):
        """Genera un audio WAV con la pronunciación nativa en inglés."""
        print("🔊 Generando pronunciación nativa con TTS...")
        temp_mp3 = "voz_temp.mp3"
        tts = gTTS(text=frase, lang="en", slow=False)
        tts.save(temp_mp3)

        if not which("ffmpeg"):
            AudioSegment.converter = os.path.join(os.getcwd(), "ffmpeg.exe")

        sound = AudioSegment.from_file(temp_mp3, format="mp3")
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(filename_wav, format="wav")
        os.remove(temp_mp3)
        print("✅ Audio de referencia creado.\n")

    # === Transcripción ===
    def transcribir_audio(self, filename):
        """Transcribe un audio a texto usando el modelo ASR."""
        speech, fs = sf.read(filename)
        input_values = self.processor(
            speech, return_tensors="pt", sampling_rate=fs
        ).input_values.to(DEVICE)
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription.lower().strip()

    # === Normalizar texto ===
    @staticmethod
    def normalizar_texto(t):
        t = t.lower()
        t = re.sub(r"[^a-z\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # === Comparar ===
    @staticmethod
    def comparar_textos(ref, user):
        return SequenceMatcher(None, ref, user).ratio()

    # === Limpiar audios temporales ===
    @staticmethod
    def limpiar_audios():
        for archivo in [REF_FILE, USER_FILE, "voz_temp.mp3"]:
            if os.path.exists(archivo):
                os.remove(archivo)

    # === PROCESO COMPLETO DE ANÁLISIS ===
    def analyze_user_pronunciation(self, user_audio_path: str, tema: str):
        """Analiza la pronunciación del usuario comparando con una referencia generada."""

        if tema not in self.temas:
            raise ValueError(f"Tema '{tema}' no reconocido. Usa uno de: {self.temas}")

        try:
            # 1️⃣ Generar oración
            frase_ref = self.generar_oracion_tema(tema)
            print(f"\n📖 Frase generada: {frase_ref}\n")

            # 2️⃣ Generar audio de referencia
            self.generar_tts(frase_ref, REF_FILE)

            # 3️⃣ Transcribir usuario
            transcripcion = self.transcribir_audio(user_audio_path)

            # 4️⃣ Comparar
            ref_norm = self.normalizar_texto(frase_ref)
            user_norm = self.normalizar_texto(transcripcion)
            similitud = self.comparar_textos(ref_norm, user_norm)
            porcentaje = round(similitud * 100, 2)

            if porcentaje >= 85:
                feedback = "Excelente pronunciación 🎉"
            elif porcentaje >= 70:
                feedback = "Buena pronunciación, pero puedes mejorar algunas palabras 👍"
            else:
                feedback = "Practica más la pronunciación y ritmo 🗣️"

            resultado = {
                "frase_ref": frase_ref,
                "transcripcion": transcripcion,
                "similitud": similitud,
                "porcentaje": porcentaje,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"🗣️ Dijiste: {transcripcion}")
            print(f"🎯 Similitud: {porcentaje}% — {feedback}\n")

            return resultado

        finally:
            self.limpiar_audios()


# ✅ Instancia global para FastAPI
pronunciation_analyzer = PronunciationAnalyzer()
