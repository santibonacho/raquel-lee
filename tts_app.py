#!/usr/bin/env python3
"""
TTS Studio - Aplicación de Texto a Voz con XTTS v2
Desarrollado con CustomTkinter para una interfaz moderna

Optimizaciones implementadas:
- Cacheo de conditioning latents (speaker embeddings)
- inference_mode() para mejor rendimiento
- Limpieza periódica de memoria (gc + mps.empty_cache)
- Parámetros de inferencia optimizados para velocidad
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import json
import threading
import platform
import datetime
import shutil
import gc
import re
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Generator
import numpy as np

# Aceptar automáticamente la licencia de Coqui TTS (CPML para uso no comercial)
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Configuración de rutas
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
VOICES_DIR = ASSETS_DIR / "voices"
OUTPUT_DIR = ASSETS_DIR / "output"
HISTORY_FILE = BASE_DIR / "history.json"

# Crear directorios si no existen
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de apariencia
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Parche para PyTorch 2.6+: permitir cargar checkpoints de XTTS v2
def _setup_torch_safe_globals():
    """Configura los globals seguros para cargar modelos XTTS (solo PyTorch 2.6+)"""
    try:
        import torch
        # Verificar si existe add_safe_globals (PyTorch 2.6+)
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
                torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])
            except ImportError:
                pass
            
            try:
                from TTS.config import BaseAudioConfig, BaseDatasetConfig
                torch.serialization.add_safe_globals([BaseAudioConfig, BaseDatasetConfig])
            except ImportError:
                pass
    except Exception:
        pass  # Ignorar si hay cualquier error


class DeviceManager:
    """Gestiona la detección y configuración del dispositivo de cómputo"""
    
    _device = None  # Cache del dispositivo
    _printed = False  # Evitar mensajes duplicados
    
    @staticmethod
    def get_device(silent=False):
        """Detecta el mejor dispositivo disponible"""
        import torch
        
        # Usar cache si ya se detectó
        if DeviceManager._device is not None:
            return DeviceManager._device
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            if torch.backends.mps.is_available():
                DeviceManager._device = "mps"
                if not silent and not DeviceManager._printed:
                    print("🍎 Usando GPU Apple Silicon (MPS)")
                    DeviceManager._printed = True
                return "mps"
        elif system == "Windows" or system == "Linux":
            if torch.cuda.is_available():
                DeviceManager._device = "cuda"
                if not silent and not DeviceManager._printed:
                    print(f"🎮 Usando GPU NVIDIA (CUDA): {torch.cuda.get_device_name(0)}")
                    DeviceManager._printed = True
                return "cuda"
        
        DeviceManager._device = "cpu"
        if not silent and not DeviceManager._printed:
            print("💻 Usando CPU (más lento)")
            DeviceManager._printed = True
        return "cpu"
    
    @staticmethod
    def is_apple_silicon():
        """Verifica si el dispositivo es Apple Silicon (MPS)"""
        return DeviceManager.get_device(silent=True) == "mps"
    
    @staticmethod
    def limpiar_memoria(dispositivo: str = None):
        """Limpia la memoria RAM y caché del dispositivo"""
        gc.collect()
        
        if dispositivo is None:
            dispositivo = DeviceManager._device
        
        if dispositivo == "mps":
            import torch
            torch.mps.empty_cache()
        elif dispositivo == "cuda":
            import torch
            torch.cuda.empty_cache()


class TTSEngine:
    """
    Motor de Text-to-Speech con Coqui TTS optimizado para Apple Silicon.
    
    Optimizaciones implementadas:
    - Cacheo de speaker embeddings (conditioning latents)
    - inference_mode() para mejor rendimiento
    - Limpieza periódica de memoria (gc + mps.empty_cache)
    - Parámetros de inferencia optimizados para velocidad
    """
    
    # Sample rate de XTTS v2
    SAMPLE_RATE = 24000
    
    # Límite de caracteres por fragmento
    MAX_CHARS_PER_CHUNK = 200
    
    # Contador para limpieza periódica de memoria
    _FRAGMENTOS_ANTES_DE_LIMPIAR = 5
    
    # Parámetros de inferencia XTTS (balance velocidad/calidad)
    XTTS_INFERENCE_PARAMS = {
        "temperature": 0.65,        # Más bajo = más determinístico y rápido
        "top_k": 30,                # Reducido de 50 para menos opciones a evaluar
        "top_p": 0.80,              # Ligeramente más restrictivo
        "repetition_penalty": 10.0,
        "length_penalty": 1.0,
        "do_sample": True,          # True para variedad natural
        "num_beams": 1,             # Sin beam search (más rápido)
        "speed": 1.0,               # Velocidad normal de reproducción
        "enable_text_splitting": False,  # Lo manejamos nosotros
    }
    
    def __init__(self):
        self.model = None
        self.device = None
        self.is_loaded = False
        self._cargando = False
        
        # Copia de parámetros de inferencia
        self.inference_params = self.XTTS_INFERENCE_PARAMS.copy()
        
        # === OPTIMIZACIÓN: Cache de Conditioning Latents ===
        self._cached_gpt_cond_latent = None
        self._cached_speaker_embedding = None
        self._cached_speaker_wav_path = None
        
        # Contador de fragmentos para limpieza de memoria
        self._fragmentos_procesados = 0
        
        # Bandera para cancelar generación
        self._cancel_generation = False
    
    def _limpiar_memoria(self, forzar: bool = False):
        """
        Limpia la memoria RAM y caché de MPS/CUDA.
        
        Args:
            forzar: Si True, limpia siempre. Si False, solo cada N fragmentos.
        """
        self._fragmentos_procesados += 1
        
        if forzar or self._fragmentos_procesados >= self._FRAGMENTOS_ANTES_DE_LIMPIAR:
            DeviceManager.limpiar_memoria(self.device)
            self._fragmentos_procesados = 0
    
    def invalidar_cache_speaker(self):
        """Invalida el cache de speaker embeddings (llamar al cambiar de voz)."""
        self._cached_gpt_cond_latent = None
        self._cached_speaker_embedding = None
        self._cached_speaker_wav_path = None
        self._limpiar_memoria(forzar=True)
    
    def _calcular_conditioning_latents(
        self, 
        speaker_wav_path: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple:
        """
        Calcula y cachea los conditioning latents del speaker.
        
        Esta es la OPTIMIZACIÓN PRINCIPAL: evita recalcular los embeddings
        del speaker en cada llamada a tts.tts().
        
        NOTA: get_conditioning_latents debe ejecutarse en CPU porque
        MPS no soporta ComplexFloat usado en los espectrogramas.
        """
        import torch
        
        # Si ya tenemos cache para este speaker, devolverla
        if (self._cached_speaker_wav_path == speaker_wav_path and 
            self._cached_gpt_cond_latent is not None and
            self._cached_speaker_embedding is not None):
            return self._cached_gpt_cond_latent, self._cached_speaker_embedding
        
        if progress_callback:
            progress_callback("Calculando embeddings de voz (en CPU)...")
        
        # Acceder al modelo XTTS subyacente
        model = self.model.synthesizer.tts_model
        config = self.model.synthesizer.tts_config
        
        # Guardar dispositivo original
        original_device = self.device
        need_cpu = original_device == "mps"
        
        if need_cpu:
            # Mover modelo completo a CPU para el cálculo de latentes
            self.model.synthesizer.tts_model.to("cpu")
        
        try:
            # Calcular latentes usando el método oficial del modelo en CPU
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=speaker_wav_path,
                    gpt_cond_len=getattr(config, 'gpt_cond_len', 30),
                    gpt_cond_chunk_len=getattr(config, 'gpt_cond_chunk_len', 6),
                    max_ref_length=getattr(config, 'max_ref_len', 30),
                )
        
        finally:
            if need_cpu:
                model.to(original_device)
        
        # Mover al dispositivo correcto
        gpt_cond_latent = gpt_cond_latent.to(original_device)
        speaker_embedding = speaker_embedding.to(original_device)
        
        # Guardar en cache memoria
        self._cached_gpt_cond_latent = gpt_cond_latent
        self._cached_speaker_embedding = speaker_embedding
        self._cached_speaker_wav_path = speaker_wav_path
        
        DeviceManager.limpiar_memoria(original_device)
        
        if progress_callback:
            progress_callback("✅ Embeddings cacheados")
        
        return gpt_cond_latent, speaker_embedding
        
    def load_model(self, progress_callback: Optional[Callable] = None):
        """
        Carga el modelo XTTS v2 con optimizaciones para Apple Silicon.
        """
        if self.is_loaded:
            return True
            
        if self._cargando:
            return False
            
        self._cargando = True
        
        try:
            import torch
            
            _setup_torch_safe_globals()
            
            # Monkey-patch para weights_only=False
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = patched_torch_load
            
            if progress_callback:
                progress_callback("Importando Coqui TTS...")
            
            from TTS.api import TTS
            
            if progress_callback:
                progress_callback("Detectando dispositivo...")
            
            self.device = DeviceManager.get_device()
            
            if progress_callback:
                progress_callback("Cargando modelo XTTS v2...")
            
            # Cargar modelo
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Mover a dispositivo
            try:
                self.model.to(self.device)
                
                # === OPTIMIZACIÓN FP16 para Apple Silicon ===
                # El M2 tiene aceleración nativa para FP16
                if DeviceManager.is_apple_silicon():
                    try:
                        # Intentar convertir a FP16 para mejor rendimiento
                        # Nota: Algunas capas pueden requerir FP32
                        # self.model.synthesizer.tts_model.half()
                        # self._use_fp16 = True
                        # print("⚡ Modelo en FP16 (Neural Engine)")
                        pass  # Desactivado por compatibilidad
                    except Exception as e:
                        print(f"⚠️ FP16 no disponible: {e}")
                        
            except Exception as e:
                print(f"⚠️ No se pudo mover a {self.device}: {e}")
                self.device = "cpu"
            
            torch.load = original_torch_load
            
            self.is_loaded = True
            self._cargando = False
            
            if progress_callback:
                progress_callback("✅ Modelo cargado y optimizado")
            
            return True
            
        except Exception as e:
            self._cargando = False
            print(f"Error cargando modelo: {e}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return False
    
    def _dividir_texto(self, texto: str, max_chars: int = None) -> List[str]:
        """Divide el texto en fragmentos procesables"""
        if max_chars is None:
            max_chars = self.MAX_CHARS_PER_CHUNK
        
        if len(texto) <= max_chars:
            return [texto.strip()]
        
        patron = r'(?<=[.!?])\s+|(?<=\n)'
        oraciones = re.split(patron, texto)
        oraciones = [s.strip() for s in oraciones if s.strip()]
        
        fragmentos = []
        fragmento_actual = ""
        
        for oracion in oraciones:
            if len(oracion) > max_chars:
                if fragmento_actual:
                    fragmentos.append(fragmento_actual.strip())
                    fragmento_actual = ""
                
                sub_partes = re.split(r'(?<=,)\s*', oracion)
                for sub in sub_partes:
                    if len(sub) > max_chars:
                        palabras = sub.split()
                        sub_fragmento = ""
                        for palabra in palabras:
                            if len(sub_fragmento) + len(palabra) + 1 <= max_chars:
                                sub_fragmento += " " + palabra if sub_fragmento else palabra
                            else:
                                if sub_fragmento:
                                    fragmentos.append(sub_fragmento.strip())
                                sub_fragmento = palabra
                        if sub_fragmento:
                            fragmentos.append(sub_fragmento.strip())
                    else:
                        fragmentos.append(sub.strip())
            elif len(fragmento_actual) + len(oracion) + 1 <= max_chars:
                fragmento_actual += " " + oracion if fragmento_actual else oracion
            else:
                if fragmento_actual:
                    fragmentos.append(fragmento_actual.strip())
                fragmento_actual = oracion
        
        if fragmento_actual:
            fragmentos.append(fragmento_actual.strip())
        
        return fragmentos
    
    def _sintetizar_fragmento(self, texto: str, gpt_cond_latent, speaker_embedding) -> Optional[np.ndarray]:
        """
        Sintetiza un fragmento con latentes pre-calculados.
        """
        import torch
        
        try:
            model = self.model.synthesizer.tts_model
            
            with torch.inference_mode():
                out = model.inference(
                    text=texto,
                    language="es",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    **self.inference_params
                )
            
            wav = out["wav"]
            
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            if isinstance(wav, list):
                wav = np.array(wav)
            
            return wav
            
        except Exception as e:
            print(f"Error sintetizando: {e}")
            return None
    
    def generate_audio_streaming(
        self, 
        text: str, 
        speaker_wav: str,
        progress_callback: Optional[Callable] = None
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        GENERADOR/STREAMING: Produce audio fragmento por fragmento.
        
        Ventajas:
        - Reduce picos de RAM (crítico en Mac 16GB)
        - Permite reproducir mientras se genera
        - La UI puede mostrar progreso real
        
        Yields:
            Tuple[np.ndarray, int]: (audio_chunk, fragment_index)
        """
        if not self.is_loaded:
            raise Exception("Modelo no cargado")
        
        # Resetear bandera de cancelación
        self._cancel_generation = False
        
        # Invalidar cache si cambió el speaker
        if self._cached_speaker_wav_path != speaker_wav:
            self.invalidar_cache_speaker()
        
        # Obtener latentes (cacheados o calcular)
        gpt_cond_latent, speaker_embedding = self._calcular_conditioning_latents(
            speaker_wav, progress_callback
        )
        
        # Dividir texto
        fragmentos = self._dividir_texto(text)
        total = len(fragmentos)
        
        if progress_callback:
            progress_callback(f"🎙️ Procesando: 0/{total} fragmentos")
        
        for i, fragmento in enumerate(fragmentos):
            # Verificar si se canceló
            if self._cancel_generation:
                if progress_callback:
                    progress_callback("❌ Generación cancelada")
                return
            
            if progress_callback:
                progress_callback(f"🎙️ Procesando: {i+1}/{total} fragmentos")
            
            wav = self._sintetizar_fragmento(fragmento, gpt_cond_latent, speaker_embedding)
            
            if wav is not None:
                yield wav, i
            
            # Limpieza periódica
            self._limpiar_memoria()
        
        # Limpieza final
        self._limpiar_memoria(forzar=True)
    
    def generate_audio(self, text: str, speaker_wav: str, output_path: str, 
                      language: str = "es", progress_callback: Optional[Callable] = None):
        """
        Genera audio completo y guarda en archivo.
        Usa el generador internamente.
        """
        try:
            if not self.is_loaded:
                raise Exception("Modelo no cargado")
            
            import soundfile as sf
            
            audios = []
            
            for wav, idx in self.generate_audio_streaming(text, speaker_wav, progress_callback):
                audios.append(wav)
            
            # Verificar si se canceló
            if self._cancel_generation:
                return False
            
            if not audios:
                raise ValueError("No se pudo generar audio")
            
            if progress_callback:
                progress_callback(f"🔗 Uniendo {len(audios)} fragmento(s)...")
            
            # Concatenar con pausas
            if len(audios) > 1:
                pausa = np.zeros(int(self.SAMPLE_RATE * 0.15))
                audio_final = []
                for i, audio in enumerate(audios):
                    audio_final.append(audio)
                    if i < len(audios) - 1:
                        audio_final.append(pausa)
                wav_final = np.concatenate(audio_final)
            else:
                wav_final = audios[0]
            
            # Guardar
            sf.write(output_path, wav_final, self.SAMPLE_RATE)
            
            if progress_callback:
                progress_callback("✅ Audio guardado")
            
            return True
            
        except Exception as e:
            print(f"Error generando audio: {e}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return False
    
    def get_conditioning_latents(self, audio_path: str, output_json: str):
        """Genera y guarda los latentes de condicionamiento para una voz"""
        try:
            if not self.is_loaded:
                return False
            
            import torch
            
            # Obtener latentes usando el modelo interno
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = self.model.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=audio_path
                )
            
            # Guardar como JSON (convertir tensores a listas)
            latents_data = {
                "gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(),
                "speaker_embedding": speaker_embedding.cpu().numpy().tolist()
            }
            
            with open(output_json, 'w') as f:
                json.dump(latents_data, f)
            
            return True
            
        except Exception as e:
            print(f"Error generando latentes: {e}")
            return False


class StreamingAudioPlayer:
    """
    Reproductor de audio que consume el generador del TTSEngine.
    
    Permite reproducir audio mientras se genera (streaming),
    ideal para textos largos donde el usuario no quiere esperar
    a que termine toda la generación.
    """
    
    SAMPLE_RATE = 24000  # XTTS v2 usa 24kHz
    
    def __init__(self):
        self._stop_flag = False
        self._playing = False
        self._current_thread: Optional[threading.Thread] = None
        self._audio_buffer: List[np.ndarray] = []
    
    def stop(self):
        """Detiene la reproducción actual"""
        self._stop_flag = True
    
    def is_playing(self) -> bool:
        """Retorna True si está reproduciendo"""
        return self._playing
    
    def play_stream(
        self,
        tts_engine: TTSEngine,
        text: str,
        speaker_wav: str,
        output_path: Optional[str] = None,
        on_start: Optional[Callable] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_chunk_ready: Optional[Callable[[np.ndarray, int], None]] = None,
        on_complete: Optional[Callable[[bool, Optional[str]], None]] = None,
    ):
        """
        Reproduce audio en streaming mientras se genera.
        
        Args:
            tts_engine: Motor TTS configurado
            text: Texto a sintetizar
            speaker_wav: Ruta al archivo WAV de referencia
            output_path: Ruta donde guardar el audio completo (opcional)
            on_start: Callback al iniciar
            on_progress: Callback de progreso (mensaje, fragmento_actual, total)
            on_chunk_ready: Callback cuando un chunk está listo (audio, índice)
            on_complete: Callback al terminar (éxito, ruta_archivo)
        """
        def _worker():
            import sounddevice as sd
            import soundfile as sf
            
            self._stop_flag = False
            self._playing = True
            self._audio_buffer = []
            
            try:
                if on_start:
                    on_start()
                
                # Contar fragmentos para progreso
                fragmentos = tts_engine._dividir_texto(text)
                total = len(fragmentos)
                
                # Generar y reproducir en streaming
                for wav, idx in tts_engine.generate_audio_streaming(text, speaker_wav):
                    if self._stop_flag:
                        break
                    
                    # Guardar en buffer para archivo final
                    self._audio_buffer.append(wav)
                    
                    # Callback de chunk
                    if on_chunk_ready:
                        on_chunk_ready(wav, idx)
                    
                    # Reproducir inmediatamente
                    sd.play(wav, self.SAMPLE_RATE)
                    sd.wait()
                    
                    # Callback de progreso
                    if on_progress:
                        on_progress(f"🎙️ Fragmento {idx+1}/{total}", idx+1, total)
                    
                    if self._stop_flag:
                        break
                
                # Guardar archivo completo si se especificó
                if output_path and self._audio_buffer and not self._stop_flag:
                    # Concatenar con pausas
                    if len(self._audio_buffer) > 1:
                        pausa = np.zeros(int(self.SAMPLE_RATE * 0.15))
                        audio_final = []
                        for i, audio in enumerate(self._audio_buffer):
                            audio_final.append(audio)
                            if i < len(self._audio_buffer) - 1:
                                audio_final.append(pausa)
                        wav_final = np.concatenate(audio_final)
                    else:
                        wav_final = self._audio_buffer[0]
                    
                    sf.write(output_path, wav_final, self.SAMPLE_RATE)
                
                if on_complete:
                    on_complete(not self._stop_flag, output_path if not self._stop_flag else None)
                    
            except Exception as e:
                print(f"Error en streaming: {e}")
                if on_complete:
                    on_complete(False, None)
            finally:
                self._playing = False
                self._stop_flag = False
        
        # Iniciar en thread separado
        self._current_thread = threading.Thread(target=_worker, daemon=True)
        self._current_thread.start()
    
    def get_combined_audio(self) -> Optional[np.ndarray]:
        """Retorna el audio combinado del buffer actual"""
        if not self._audio_buffer:
            return None
        
        if len(self._audio_buffer) > 1:
            pausa = np.zeros(int(self.SAMPLE_RATE * 0.15))
            audio_final = []
            for i, audio in enumerate(self._audio_buffer):
                audio_final.append(audio)
                if i < len(self._audio_buffer) - 1:
                    audio_final.append(pausa)
            return np.concatenate(audio_final)
        else:
            return self._audio_buffer[0]


class TextExtractor:
    """Extrae texto de diferentes formatos de archivo"""
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extrae texto de archivo .txt"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extrae texto de archivo .pdf usando PyMuPDF"""
        import fitz  # PyMuPDF
        
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extrae texto de archivo .docx"""
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    
    @staticmethod
    def extract_from_image(file_path: str) -> str:
        """Extrae texto de imagen usando OCR (pytesseract)"""
        import pytesseract
        from PIL import Image
        
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='spa+eng')
        return text.strip()
    
    @staticmethod
    def extract(file_path: str) -> str:
        """Extrae texto según la extensión del archivo"""
        ext = Path(file_path).suffix.lower()
        
        extractors = {
            '.txt': TextExtractor.extract_from_txt,
            '.pdf': TextExtractor.extract_from_pdf,
            '.docx': TextExtractor.extract_from_docx,
            '.doc': TextExtractor.extract_from_docx,
            '.png': TextExtractor.extract_from_image,
            '.jpg': TextExtractor.extract_from_image,
            '.jpeg': TextExtractor.extract_from_image,
            '.bmp': TextExtractor.extract_from_image,
            '.tiff': TextExtractor.extract_from_image,
        }
        
        if ext in extractors:
            return extractors[ext](file_path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")


class AudioRecorder:
    """Grabador de audio para muestras de voz"""
    
    def __init__(self, duration: int = 6, sample_rate: int = 22050):
        self.duration = duration
        self.sample_rate = sample_rate
        self.recording = None
        self.is_recording = False
    
    def record(self, progress_callback: Optional[Callable] = None):
        """Graba audio desde el micrófono"""
        import sounddevice as sd
        
        try:
            if progress_callback:
                progress_callback(f"Grabando {self.duration} segundos...")
            
            self.is_recording = True
            self.recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            self.is_recording = False
            
            if progress_callback:
                progress_callback("Grabación completada")
            
            return True
            
        except Exception as e:
            self.is_recording = False
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return False
    
    def save(self, output_path: str):
        """Guarda la grabación como archivo WAV"""
        import soundfile as sf
        
        if self.recording is not None:
            sf.write(output_path, self.recording, self.sample_rate)
            return True
        return False


class HistoryManager:
    """Gestiona el historial de generaciones"""
    
    def __init__(self, history_file: Path = HISTORY_FILE):
        self.history_file = history_file
        self.history = self._load()
    
    def _load(self) -> list:
        """Carga el historial desde archivo"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save(self):
        """Guarda el historial en archivo"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def add_entry(self, voice: str, text: str, audio_path: str, duration: float):
        """Añade una entrada al historial"""
        entry = {
            "id": len(self.history) + 1,
            "date": datetime.datetime.now().isoformat(),
            "voice": voice,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "full_text": text,
            "audio_path": audio_path,
            "duration": round(duration, 2)
        }
        self.history.append(entry)
        self._save()
        return entry
    
    def delete_entry(self, entry_id: int):
        """Elimina una entrada del historial"""
        for i, entry in enumerate(self.history):
            if entry["id"] == entry_id:
                # Eliminar archivo de audio si existe
                audio_path = Path(entry["audio_path"])
                if audio_path.exists():
                    audio_path.unlink()
                
                self.history.pop(i)
                self._save()
                return True
        return False
    
    def get_all(self) -> list:
        """Retorna todo el historial"""
        return self.history


class VoiceManager:
    """Gestiona las voces disponibles"""
    
    def __init__(self, voices_dir: Path = VOICES_DIR):
        self.voices_dir = voices_dir
    
    def get_voices(self) -> list:
        """Retorna lista de voces disponibles"""
        voices = []
        if self.voices_dir.exists():
            for folder in self.voices_dir.iterdir():
                if folder.is_dir():
                    # Buscar archivo de audio de referencia
                    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
                    if audio_files:
                        info_file = folder / "info.json"
                        info = {}
                        if info_file.exists():
                            with open(info_file, 'r', encoding='utf-8') as f:
                                info = json.load(f)
                        
                        voices.append({
                            "name": folder.name,
                            "path": str(folder),
                            "audio": str(audio_files[0]),
                            "description": info.get("description", ""),
                            "has_latents": (folder / "latents.json").exists()
                        })
        return voices
    
    def create_voice(self, name: str, description: str, audio_path: str) -> bool:
        """Crea una nueva voz"""
        try:
            voice_folder = self.voices_dir / name
            voice_folder.mkdir(parents=True, exist_ok=True)
            
            # Copiar o mover archivo de audio
            audio_dest = voice_folder / f"reference{Path(audio_path).suffix}"
            shutil.copy2(audio_path, audio_dest)
            
            # Guardar información
            info = {
                "name": name,
                "description": description,
                "created": datetime.datetime.now().isoformat()
            }
            with open(voice_folder / "info.json", 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error creando voz: {e}")
            return False
    
    def delete_voice(self, name: str) -> bool:
        """Elimina una voz"""
        try:
            voice_folder = self.voices_dir / name
            if voice_folder.exists():
                shutil.rmtree(voice_folder)
                return True
            return False
        except Exception as e:
            print(f"Error eliminando voz: {e}")
            return False


# ============== INTERFAZ GRÁFICA ==============

class StudioTab(ctk.CTkFrame):
    """Pestaña de Estudio - Zona de Generación"""
    
    def __init__(self, parent, tts_engine: TTSEngine, voice_manager: VoiceManager, 
                 history_manager: HistoryManager):
        super().__init__(parent)
        
        self.tts_engine = tts_engine
        self.voice_manager = voice_manager
        self.history_manager = history_manager
        
        self.setup_ui()
        self.refresh_voices()
    
    def setup_ui(self):
        """Configura la interfaz del estudio"""
        # Configurar grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # === Selector de Voz (Arriba) ===
        voice_frame = ctk.CTkFrame(self)
        voice_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        voice_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(voice_frame, text="🎤 Voz:", font=("", 16, "bold")).grid(
            row=0, column=0, padx=(15, 10), pady=15
        )
        
        self.voice_combo = ctk.CTkComboBox(
            voice_frame, 
            values=["Cargando..."],
            font=("", 14),
            height=40,
            width=300
        )
        self.voice_combo.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        
        refresh_btn = ctk.CTkButton(
            voice_frame, 
            text="⟳", 
            width=40,
            command=self.refresh_voices
        )
        refresh_btn.grid(row=0, column=2, padx=(5, 15), pady=15)
        
        # === Área de Contenido (Centro) ===
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Barra de herramientas
        toolbar = ctk.CTkFrame(content_frame, fg_color="transparent")
        toolbar.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        
        ctk.CTkLabel(toolbar, text="📝 Texto a convertir:", font=("", 14, "bold")).pack(
            side="left", padx=5
        )
        
        import_btn = ctk.CTkButton(
            toolbar,
            text="📁 Importar Archivo",
            command=self.import_file,
            width=150
        )
        import_btn.pack(side="right", padx=5)
        
        clear_btn = ctk.CTkButton(
            toolbar,
            text="🗑 Limpiar",
            command=self.clear_text,
            width=100,
            fg_color="gray40"
        )
        clear_btn.pack(side="right", padx=5)
        
        # Área de texto
        self.text_input = ctk.CTkTextbox(
            content_frame,
            font=("", 14),
            wrap="word"
        )
        self.text_input.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.text_input.insert("0.0", "Escribe aquí el texto que deseas convertir a voz...")
        
        # === Controles (Abajo) ===
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Barra de progreso
        self.progress_bar = ctk.CTkProgressBar(control_frame, mode="indeterminate")
        self.progress_bar.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
        self.progress_bar.set(0)
        
        # Label de estado
        self.status_label = ctk.CTkLabel(
            control_frame, 
            text="Listo para generar",
            font=("", 12)
        )
        self.status_label.grid(row=1, column=0, padx=15, pady=5)
        
        # Botón de generar
        self.generate_btn = ctk.CTkButton(
            control_frame,
            text="🔊 GENERAR AUDIO",
            font=("", 18, "bold"),
            height=50,
            command=self.generate_audio
        )
        self.generate_btn.grid(row=2, column=0, padx=15, pady=(10, 15), sticky="ew")
    
    def refresh_voices(self):
        """Actualiza la lista de voces"""
        voices = self.voice_manager.get_voices()
        voice_names = [v["name"] for v in voices]
        
        if voice_names:
            self.voice_combo.configure(values=voice_names)
            self.voice_combo.set(voice_names[0])
        else:
            self.voice_combo.configure(values=["Sin voces disponibles"])
            self.voice_combo.set("Sin voces disponibles")
    
    def import_file(self):
        """Importa texto desde un archivo"""
        filetypes = [
            ("Todos los soportados", "*.txt *.pdf *.docx *.doc *.png *.jpg *.jpeg"),
            ("Texto", "*.txt"),
            ("PDF", "*.pdf"),
            ("Word", "*.docx *.doc"),
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")
        ]
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if file_path:
            self.status_label.configure(text="Extrayendo texto...")
            self.progress_bar.start()
            
            def extract():
                try:
                    text = TextExtractor.extract(file_path)
                    self.after(0, lambda: self._insert_extracted_text(text))
                except Exception as e:
                    self.after(0, lambda: self._show_error(f"Error extrayendo texto: {e}"))
            
            threading.Thread(target=extract, daemon=True).start()
    
    def _insert_extracted_text(self, text: str):
        """Inserta el texto extraído en el área de texto"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.text_input.delete("0.0", "end")
        self.text_input.insert("0.0", text)
        self.status_label.configure(text="Texto importado correctamente")
    
    def _show_error(self, message: str):
        """Muestra un error"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.status_label.configure(text=message)
        messagebox.showerror("Error", message)
    
    def clear_text(self):
        """Limpia el área de texto"""
        self.text_input.delete("0.0", "end")
    
    def generate_audio(self):
        """Genera el audio con soporte de streaming"""
        text = self.text_input.get("0.0", "end").strip()
        voice_name = self.voice_combo.get()
        
        if not text or text == "Escribe aquí el texto que deseas convertir a voz...":
            messagebox.showwarning("Aviso", "Por favor, ingresa el texto a convertir")
            return
        
        if voice_name == "Sin voces disponibles":
            messagebox.showwarning("Aviso", "No hay voces disponibles. Crea una en 'Mis Voces'")
            return
        
        # Obtener ruta del audio de referencia
        voices = self.voice_manager.get_voices()
        voice = next((v for v in voices if v["name"] == voice_name), None)
        
        if not voice:
            messagebox.showerror("Error", "Voz no encontrada")
            return
        
        # Preparar generación
        self.generate_btn.configure(state="disabled", text="⏹ DETENER")
        self.generate_btn.configure(command=self._stop_generation)
        self.progress_bar.start()
        self.status_label.configure(text="Generando audio...")
        
        # Nombre del archivo de salida
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"{voice_name}_{timestamp}.wav")
        
        # Siempre usar generación completa (sin streaming de reproducción)
        # Se genera todo primero y luego se reproduce al final
        
        def generate_normal():
            """Generación normal (sin streaming)"""
            try:
                # Verificar que el modelo está cargado
                if not self.tts_engine.is_loaded:
                    self.tts_engine.load_model(
                        progress_callback=lambda msg: self.after(
                            0, lambda m=msg: self.status_label.configure(text=m)
                        )
                    )
                
                success = self.tts_engine.generate_audio(
                    text=text,
                    speaker_wav=voice["audio"],
                    output_path=output_path,
                    language="es",
                    progress_callback=lambda msg: self.after(
                        0, lambda m=msg: self.status_label.configure(text=m)
                    )
                )
                
                if success:
                    # Obtener duración del audio
                    import soundfile as sf
                    info = sf.info(output_path)
                    duration = info.duration
                    
                    # Añadir al historial
                    self.history_manager.add_entry(voice_name, text, output_path, duration)
                    
                    self.after(0, lambda: self._generation_complete(output_path))
                else:
                    self.after(0, lambda: self._generation_failed("Error en la generación"))
                    
            except Exception as e:
                self.after(0, lambda: self._generation_failed(str(e)))
        
        # Iniciar generación en thread separado
        threading.Thread(target=generate_normal, daemon=True).start()
    
    def _stop_generation(self):
        """Detiene la generación en curso"""
        # Señalizar al engine que cancele
        self.tts_engine._cancel_generation = True
        self.generate_btn.configure(state="normal", text="🔊 GENERAR AUDIO")
        self.generate_btn.configure(command=self.generate_audio)
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.status_label.configure(text="Generación cancelada")
    
    def _generation_complete(self, output_path: str):
        """Callback cuando la generación termina"""
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.generate_btn.configure(state="normal", text="🔊 GENERAR AUDIO")
        self.generate_btn.configure(command=self.generate_audio)
        self.status_label.configure(text=f"✓ Audio guardado: {Path(output_path).name}")
        
        # Preguntar si reproducir
        if messagebox.askyesno("Éxito", "Audio generado correctamente.\n¿Deseas reproducirlo?"):
            self.play_audio(output_path)
    
    def _generation_failed(self, error: str):
        """Callback cuando la generación falla"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.generate_btn.configure(state="normal", text="🔊 GENERAR AUDIO")
        self.generate_btn.configure(command=self.generate_audio)
        self.status_label.configure(text=f"✗ Error: {error}")
        messagebox.showerror("Error", f"Error generando audio:\n{error}")
    
    def play_audio(self, audio_path: str):
        """Reproduce un archivo de audio"""
        try:
            import soundfile as sf
            import sounddevice as sd
            
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
        except Exception as e:
            messagebox.showerror("Error", f"Error reproduciendo audio: {e}")


class VoicesTab(ctk.CTkFrame):
    """Pestaña de Mis Voces - Gestor de Speakers"""
    
    def __init__(self, parent, tts_engine: TTSEngine, voice_manager: VoiceManager):
        super().__init__(parent)
        
        self.tts_engine = tts_engine
        self.voice_manager = voice_manager
        self.sample_duration = 6  # Duración configurable (6-15 segundos)
        self.recorder = AudioRecorder(duration=self.sample_duration)
        self.temp_audio_path = None
        
        self.setup_ui()
        self.refresh_voices_list()
    
    def setup_ui(self):
        """Configura la interfaz"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # === Panel izquierdo: Lista de voces ===
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(left_frame, text="🎭 Voces Instaladas", font=("", 18, "bold")).grid(
            row=0, column=0, padx=15, pady=15
        )
        
        # Lista scrollable de voces
        self.voices_scroll = ctk.CTkScrollableFrame(left_frame)
        self.voices_scroll.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.voices_scroll.grid_columnconfigure(0, weight=1)
        
        # === Panel derecho: Crear nueva voz ===
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(right_frame, text="➕ Crear Nueva Voz", font=("", 18, "bold")).grid(
            row=0, column=0, padx=15, pady=15
        )
        
        # Formulario
        form_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        form_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        form_frame.grid_columnconfigure(1, weight=1)
        
        # Nombre
        ctk.CTkLabel(form_frame, text="Nombre:", font=("", 14)).grid(
            row=0, column=0, padx=5, pady=10, sticky="w"
        )
        self.name_entry = ctk.CTkEntry(form_frame, placeholder_text="Ej: Narrador Español")
        self.name_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        # Descripción
        ctk.CTkLabel(form_frame, text="Descripción:", font=("", 14)).grid(
            row=1, column=0, padx=5, pady=10, sticky="w"
        )
        self.desc_entry = ctk.CTkEntry(form_frame, placeholder_text="Ej: Voz masculina grave")
        self.desc_entry.grid(row=1, column=1, padx=5, pady=10, sticky="ew")
        
        # Audio de referencia
        audio_frame = ctk.CTkFrame(right_frame)
        audio_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        audio_frame.grid_columnconfigure(0, weight=1)
        audio_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(audio_frame, text="Audio de Referencia:", font=("", 14, "bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(15, 10)
        )
        
        # Slider para configurar duración de grabación
        duration_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        duration_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="ew")
        duration_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(duration_frame, text="Duración:", font=("", 12)).grid(
            row=0, column=0, padx=(0, 10), sticky="w"
        )
        
        self.duration_slider = ctk.CTkSlider(
            duration_frame,
            from_=6,
            to=15,
            number_of_steps=9,
            command=self._on_duration_change
        )
        self.duration_slider.set(6)
        self.duration_slider.grid(row=0, column=1, padx=5, sticky="ew")
        
        self.duration_label = ctk.CTkLabel(duration_frame, text="6 seg", font=("", 12, "bold"))
        self.duration_label.grid(row=0, column=2, padx=(10, 0), sticky="e")
        
        self.record_btn = ctk.CTkButton(
            audio_frame,
            text="🎙 Grabar (6 seg)",
            command=self.record_sample,
            height=40
        )
        self.record_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        self.upload_btn = ctk.CTkButton(
            audio_frame,
            text="📁 Subir Audio",
            command=self.upload_wav,
            height=40,
            fg_color="gray40"
        )
        self.upload_btn.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        
        # Estado del audio
        self.audio_status = ctk.CTkLabel(
            audio_frame, 
            text="Sin audio seleccionado",
            font=("", 12),
            text_color="gray"
        )
        self.audio_status.grid(row=3, column=0, columnspan=2, padx=10, pady=(5, 15))
        
        # Progreso
        self.progress_bar = ctk.CTkProgressBar(right_frame, mode="indeterminate")
        self.progress_bar.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(right_frame, text="", font=("", 12))
        self.status_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Botón guardar
        self.save_btn = ctk.CTkButton(
            right_frame,
            text="💾 GUARDAR VOZ",
            font=("", 16, "bold"),
            height=50,
            command=self.save_voice
        )
        self.save_btn.grid(row=5, column=0, padx=20, pady=20, sticky="ew")
    
    def refresh_voices_list(self):
        """Actualiza la lista de voces"""
        # Limpiar lista actual
        for widget in self.voices_scroll.winfo_children():
            widget.destroy()
        
        voices = self.voice_manager.get_voices()
        
        if not voices:
            ctk.CTkLabel(
                self.voices_scroll, 
                text="No hay voces instaladas.\nCrea una nueva usando el panel derecho.",
                font=("", 14),
                text_color="gray"
            ).pack(pady=50)
            return
        
        for voice in voices:
            self._create_voice_card(voice)
    
    def _create_voice_card(self, voice: dict):
        """Crea una tarjeta para una voz"""
        card = ctk.CTkFrame(self.voices_scroll)
        card.pack(fill="x", padx=5, pady=5)
        card.grid_columnconfigure(1, weight=1)
        
        # Icono
        ctk.CTkLabel(card, text="🎤", font=("", 24)).grid(
            row=0, column=0, rowspan=2, padx=15, pady=10
        )
        
        # Nombre
        ctk.CTkLabel(card, text=voice["name"], font=("", 14, "bold")).grid(
            row=0, column=1, padx=5, pady=(10, 0), sticky="w"
        )
        
        # Descripción
        desc = voice.get("description", "Sin descripción")
        ctk.CTkLabel(card, text=desc, font=("", 12), text_color="gray").grid(
            row=1, column=1, padx=5, pady=(0, 10), sticky="w"
        )
        
        # Botones
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10)
        
        play_btn = ctk.CTkButton(
            btn_frame,
            text="▶",
            width=35,
            command=lambda v=voice: self.play_voice_sample(v)
        )
        play_btn.pack(side="left", padx=2)
        
        delete_btn = ctk.CTkButton(
            btn_frame,
            text="🗑",
            width=35,
            fg_color="red",
            hover_color="darkred",
            command=lambda v=voice: self.delete_voice(v)
        )
        delete_btn.pack(side="left", padx=2)
    
    def _on_duration_change(self, value):
        """Callback cuando cambia el slider de duración"""
        duration = int(value)
        self.sample_duration = duration
        self.recorder = AudioRecorder(duration=duration)
        self.duration_label.configure(text=f"{duration} seg")
        self.record_btn.configure(text=f"🎙 Grabar ({duration} seg)")
    
    def play_voice_sample(self, voice: dict):
        """Reproduce la muestra de una voz"""
        try:
            import soundfile as sf
            import sounddevice as sd
            
            data, samplerate = sf.read(voice["audio"])
            sd.play(data, samplerate)
        except Exception as e:
            messagebox.showerror("Error", f"Error reproduciendo: {e}")
    
    def delete_voice(self, voice: dict):
        """Elimina una voz"""
        if messagebox.askyesno("Confirmar", f"¿Eliminar la voz '{voice['name']}'?"):
            if self.voice_manager.delete_voice(voice["name"]):
                self.refresh_voices_list()
                messagebox.showinfo("Éxito", "Voz eliminada")
            else:
                messagebox.showerror("Error", "No se pudo eliminar la voz")
    
    def record_sample(self):
        """Graba una muestra de voz"""
        self.record_btn.configure(state="disabled", text="🔴 Grabando...")
        self.progress_bar.start()
        self.status_label.configure(text=f"Grabando {self.sample_duration} segundos...")
        
        def record():
            success = self.recorder.record()
            
            if success:
                # Guardar temporalmente
                self.temp_audio_path = str(BASE_DIR / "temp_recording.wav")
                self.recorder.save(self.temp_audio_path)
                
                self.after(0, lambda: self._recording_complete())
            else:
                self.after(0, lambda: self._recording_failed())
        
        threading.Thread(target=record, daemon=True).start()
    
    def _recording_complete(self):
        """Callback cuando la grabación termina"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.record_btn.configure(state="normal", text=f"🎙 Grabar ({self.sample_duration} seg)")
        self.audio_status.configure(text="✓ Audio grabado", text_color="green")
        self.status_label.configure(text="Grabación lista")
    
    def _recording_failed(self):
        """Callback cuando la grabación falla"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.record_btn.configure(state="normal", text=f"🎙 Grabar ({self.sample_duration} seg)")
        self.audio_status.configure(text="✗ Error en grabación", text_color="red")
        self.status_label.configure(text="")
    
    def upload_wav(self):
        """Sube un archivo WAV, MP3 u OPUS"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Archivos de Audio", "*.wav *.mp3 *.opus"),
                ("Archivos WAV", "*.wav"),
                ("Archivos MP3", "*.mp3"),
                ("Archivos OPUS", "*.opus"),
                ("Todos", "*.*")
            ]
        )
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in (".mp3", ".opus"):
                # Convertir MP3/OPUS a WAV y cortar a la duración seleccionada
                self.upload_btn.configure(state="disabled")
                self.progress_bar.start()
                self.status_label.configure(text=f"Convirtiendo {file_ext.upper()} a WAV...")
                
                def convert():
                    try:
                        converted_path = self._convert_audio_to_wav(file_path, file_ext)
                        self.after(0, lambda: self._upload_complete(converted_path, Path(file_path).name))
                    except Exception as e:
                        self.after(0, lambda: self._upload_failed(str(e)))
                
                threading.Thread(target=convert, daemon=True).start()
            elif file_ext == ".wav":
                # Archivo WAV: cortar a la duración seleccionada
                self.upload_btn.configure(state="disabled")
                self.progress_bar.start()
                self.status_label.configure(text="Procesando audio...")
                
                def process():
                    try:
                        processed_path = self._process_wav_duration(file_path)
                        self.after(0, lambda: self._upload_complete(processed_path, Path(file_path).name))
                    except Exception as e:
                        self.after(0, lambda: self._upload_failed(str(e)))
                
                threading.Thread(target=process, daemon=True).start()
    
    def _convert_audio_to_wav(self, audio_path: str, file_ext: str) -> str:
        """Convierte un archivo de audio (MP3, OPUS) a WAV y lo corta a la duración configurada"""
        from pydub import AudioSegment
        
        # Cargar el audio según su formato
        if file_ext == ".mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif file_ext == ".opus":
            audio = AudioSegment.from_ogg(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)
        
        # Cortar a la duración configurada (en milisegundos)
        max_duration_ms = self.sample_duration * 1000
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
        
        # Convertir a mono y sample rate adecuado
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(22050)
        
        # Guardar como WAV temporal
        output_path = str(BASE_DIR / "temp_converted.wav")
        audio.export(output_path, format="wav")
        
        return output_path
    
    def _process_wav_duration(self, wav_path: str) -> str:
        """Procesa un archivo WAV y lo corta a la duración configurada"""
        from pydub import AudioSegment
        
        # Cargar el WAV
        audio = AudioSegment.from_wav(wav_path)
        
        # Cortar a la duración configurada (en milisegundos)
        max_duration_ms = self.sample_duration * 1000
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
            
            # Convertir a mono y sample rate adecuado
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(22050)
            
            # Guardar como WAV temporal
            output_path = str(BASE_DIR / "temp_processed.wav")
            audio.export(output_path, format="wav")
            return output_path
        
        # Si ya está dentro de la duración, usar el archivo original
        return wav_path
    
    def _upload_complete(self, audio_path: str, original_name: str):
        """Callback cuando la subida/conversión termina"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.upload_btn.configure(state="normal")
        self.temp_audio_path = audio_path
        self.audio_status.configure(
            text=f"✓ {original_name} (máx {self.sample_duration}s)", 
            text_color="green"
        )
        self.status_label.configure(text="Audio listo")
    
    def _upload_failed(self, error: str):
        """Callback cuando la subida/conversión falla"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.upload_btn.configure(state="normal")
        self.audio_status.configure(text=f"✗ Error: {error}", text_color="red")
        self.status_label.configure(text="")
    
    def save_voice(self):
        """Guarda la nueva voz"""
        name = self.name_entry.get().strip()
        description = self.desc_entry.get().strip()
        
        if not name:
            messagebox.showwarning("Aviso", "Por favor, ingresa un nombre para la voz")
            return
        
        if not self.temp_audio_path or not Path(self.temp_audio_path).exists():
            messagebox.showwarning("Aviso", "Por favor, graba o sube un audio de referencia")
            return
        
        # Validar que el nombre no exista
        existing = [v["name"].lower() for v in self.voice_manager.get_voices()]
        if name.lower() in existing:
            messagebox.showwarning("Aviso", "Ya existe una voz con ese nombre")
            return
        
        self.save_btn.configure(state="disabled")
        self.progress_bar.start()
        self.status_label.configure(text="Guardando voz...")
        
        def save():
            try:
                success = self.voice_manager.create_voice(name, description, self.temp_audio_path)
                
                if success:
                    # Intentar generar latentes si el modelo está cargado
                    if self.tts_engine.is_loaded:
                        voice_folder = VOICES_DIR / name
                        audio_file = list(voice_folder.glob("reference.*"))[0]
                        latents_file = voice_folder / "latents.json"
                        self.tts_engine.get_conditioning_latents(str(audio_file), str(latents_file))
                    
                    self.after(0, lambda: self._save_complete())
                else:
                    self.after(0, lambda: self._save_failed("Error guardando voz"))
                    
            except Exception as e:
                self.after(0, lambda: self._save_failed(str(e)))
        
        threading.Thread(target=save, daemon=True).start()
    
    def _save_complete(self):
        """Callback cuando se guarda la voz"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.save_btn.configure(state="normal")
        self.status_label.configure(text="✓ Voz guardada")
        
        # Limpiar formulario
        self.name_entry.delete(0, "end")
        self.desc_entry.delete(0, "end")
        self.temp_audio_path = None
        self.audio_status.configure(text="Sin audio seleccionado", text_color="gray")
        
        # Refrescar lista
        self.refresh_voices_list()
        
        # Limpiar archivo temporal
        temp_file = BASE_DIR / "temp_recording.wav"
        if temp_file.exists():
            temp_file.unlink()
        
        messagebox.showinfo("Éxito", "Voz creada correctamente")
    
    def _save_failed(self, error: str):
        """Callback cuando falla el guardado"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.save_btn.configure(state="normal")
        self.status_label.configure(text=f"✗ Error: {error}")
        messagebox.showerror("Error", f"Error guardando voz:\n{error}")


class LibraryTab(ctk.CTkFrame):
    """Pestaña de Biblioteca - Historial de generaciones"""
    
    def __init__(self, parent, history_manager: HistoryManager):
        super().__init__(parent)
        
        self.history_manager = history_manager
        self.setup_ui()
        self.refresh_history()
    
    def setup_ui(self):
        """Configura la interfaz"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(header, text="📚 Biblioteca de Audio", font=("", 20, "bold")).pack(
            side="left"
        )
        
        refresh_btn = ctk.CTkButton(
            header,
            text="⟳ Actualizar",
            width=120,
            command=self.refresh_history
        )
        refresh_btn.pack(side="right")
        
        # Lista scrollable
        self.history_scroll = ctk.CTkScrollableFrame(self)
        self.history_scroll.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.history_scroll.grid_columnconfigure(0, weight=1)
    
    def refresh_history(self):
        """Actualiza el historial"""
        # Limpiar lista actual
        for widget in self.history_scroll.winfo_children():
            widget.destroy()
        
        history = self.history_manager.get_all()
        
        if not history:
            ctk.CTkLabel(
                self.history_scroll,
                text="No hay generaciones en el historial.\nGenera audio desde la pestaña 'Estudio'.",
                font=("", 14),
                text_color="gray"
            ).pack(pady=50)
            return
        
        # Mostrar en orden inverso (más reciente primero)
        for entry in reversed(history):
            self._create_history_card(entry)
    
    def _create_history_card(self, entry: dict):
        """Crea una tarjeta para una entrada del historial"""
        card = ctk.CTkFrame(self.history_scroll)
        card.pack(fill="x", padx=5, pady=5)
        card.grid_columnconfigure(1, weight=1)
        
        # Info
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="ew")
        info_frame.grid_columnconfigure(1, weight=1)
        
        # Fecha y voz
        date_str = datetime.datetime.fromisoformat(entry["date"]).strftime("%d/%m/%Y %H:%M")
        ctk.CTkLabel(info_frame, text=f"📅 {date_str}", font=("", 12)).pack(side="left", padx=5)
        ctk.CTkLabel(info_frame, text=f"🎤 {entry['voice']}", font=("", 12, "bold")).pack(
            side="left", padx=15
        )
        ctk.CTkLabel(info_frame, text=f"⏱ {entry['duration']}s", font=("", 12)).pack(side="right", padx=5)
        
        # Texto preview
        text_frame = ctk.CTkFrame(card, fg_color="gray20", corner_radius=8)
        text_frame.grid(row=1, column=0, columnspan=2, padx=15, pady=5, sticky="ew")
        
        ctk.CTkLabel(
            text_frame,
            text=entry["text_preview"],
            font=("", 12),
            wraplength=600,
            justify="left"
        ).pack(padx=10, pady=10, anchor="w")
        
        # Botones
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, padx=15, pady=(5, 15), sticky="e")
        
        play_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Reproducir",
            width=100,
            command=lambda e=entry: self.play_audio(e)
        )
        play_btn.pack(side="left", padx=5)
        
        delete_btn = ctk.CTkButton(
            btn_frame,
            text="🗑 Eliminar",
            width=100,
            fg_color="red",
            hover_color="darkred",
            command=lambda e=entry: self.delete_entry(e)
        )
        delete_btn.pack(side="left", padx=5)
    
    def play_audio(self, entry: dict):
        """Reproduce un audio del historial"""
        audio_path = entry["audio_path"]
        
        if not Path(audio_path).exists():
            messagebox.showerror("Error", "El archivo de audio no existe")
            return
        
        try:
            import soundfile as sf
            import sounddevice as sd
            
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
        except Exception as e:
            messagebox.showerror("Error", f"Error reproduciendo: {e}")
    
    def delete_entry(self, entry: dict):
        """Elimina una entrada del historial"""
        if messagebox.askyesno("Confirmar", "¿Eliminar esta generación?"):
            if self.history_manager.delete_entry(entry["id"]):
                self.refresh_history()
            else:
                messagebox.showerror("Error", "No se pudo eliminar")


class TTSApp(ctk.CTk):
    """Aplicación principal de TTS"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración de ventana
        self.title("🎙 TTS Studio - XTTS v2")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Inicializar managers
        self.tts_engine = TTSEngine()
        self.voice_manager = VoiceManager()
        self.history_manager = HistoryManager()
        
        self.setup_ui()
        
        # Cargar modelo en segundo plano
        self.load_model_async()
    
    def setup_ui(self):
        """Configura la interfaz principal"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            header, 
            text="🎙 TTS Studio",
            font=("", 24, "bold")
        ).grid(row=0, column=0, padx=20, pady=15)
        
        self.model_status = ctk.CTkLabel(
            header,
            text="⏳ Cargando modelo...",
            font=("", 12),
            text_color="orange"
        )
        self.model_status.grid(row=0, column=2, padx=20, pady=15)
        
        # Tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        
        # Crear pestañas
        self.tabview.add("🎬 Estudio")
        self.tabview.add("🎭 Mis Voces")
        self.tabview.add("📚 Biblioteca")
        
        # Configurar cada pestaña
        studio_tab = self.tabview.tab("🎬 Estudio")
        studio_tab.grid_columnconfigure(0, weight=1)
        studio_tab.grid_rowconfigure(0, weight=1)
        
        self.studio = StudioTab(
            studio_tab, 
            self.tts_engine, 
            self.voice_manager, 
            self.history_manager
        )
        self.studio.grid(row=0, column=0, sticky="nsew")
        
        voices_tab = self.tabview.tab("🎭 Mis Voces")
        voices_tab.grid_columnconfigure(0, weight=1)
        voices_tab.grid_rowconfigure(0, weight=1)
        
        self.voices = VoicesTab(voices_tab, self.tts_engine, self.voice_manager)
        self.voices.grid(row=0, column=0, sticky="nsew")
        
        library_tab = self.tabview.tab("📚 Biblioteca")
        library_tab.grid_columnconfigure(0, weight=1)
        library_tab.grid_rowconfigure(0, weight=1)
        
        self.library = LibraryTab(library_tab, self.history_manager)
        self.library.grid(row=0, column=0, sticky="nsew")
        
        # Vincular cambio de pestaña para refrescar datos
        self.tabview.configure(command=self.on_tab_change)
    
    def on_tab_change(self):
        """Callback cuando cambia la pestaña"""
        current = self.tabview.get()
        
        if current == "🎬 Estudio":
            self.studio.refresh_voices()
        elif current == "🎭 Mis Voces":
            self.voices.refresh_voices_list()
        elif current == "📚 Biblioteca":
            self.library.refresh_history()
    
    def load_model_async(self):
        """Carga el modelo en segundo plano"""
        def load():
            success = self.tts_engine.load_model(
                progress_callback=lambda msg: self.after(
                    0, lambda m=msg: self.model_status.configure(text=f"⏳ {m}")
                )
            )
            
            if success:
                self.after(0, lambda: self.model_status.configure(
                    text="✓ Modelo cargado",
                    text_color="green"
                ))
            else:
                self.after(0, lambda: self.model_status.configure(
                    text="✗ Error cargando modelo",
                    text_color="red"
                ))
        
        threading.Thread(target=load, daemon=True).start()


def main():
    """Punto de entrada principal"""
    app = TTSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
