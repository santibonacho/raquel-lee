# 🎙 TTS Studio - Aplicación de Texto a Voz con XTTS v2

Una aplicación de escritorio profesional para convertir texto a voz utilizando el modelo XTTS v2 de Coqui TTS con una interfaz moderna construida con CustomTkinter.

## ✨ Características

- **Interfaz moderna** con 3 pestañas: Estudio, Mis Voces y Biblioteca
- **Clonación de voz** usando XTTS v2
- **Importación de archivos**: TXT, PDF, DOCX e imágenes (OCR)
- **Gestor de voces**: Graba o sube muestras de audio para crear voces personalizadas
- **Historial completo** de todas las generaciones
- **Optimización hardware**: Usa MPS en macOS, CUDA en Windows/Linux
- **Interfaz no bloqueante**: Todas las operaciones pesadas en hilos separados

## 📋 Requisitos Previos

### macOS

1. **Homebrew** (si no lo tienes):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Tesseract** (para OCR en imágenes):
   ```bash
   brew install tesseract
   brew install tesseract-lang  # Idiomas adicionales (opcional)
   ```

3. **Python 3.9+** (recomendado 3.10 o 3.11):
   ```bash
   brew install python@3.11
   ```

### Windows

1. **Tesseract OCR**:
   - Descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki
   - Añade la ruta de instalación al PATH del sistema

2. **Python 3.9+** desde python.org

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-spa python3-pip
```

## 🚀 Instalación

1. **Clona o descarga el proyecto**:
   ```bash
   cd /ruta/al/proyecto
   ```

2. **Crea un entorno virtual** (recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # o en Windows:
   # venv\Scripts\activate
   ```

3. **Instala las dependencias**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Primera ejecución** (descargará el modelo XTTS v2, ~2GB):
   ```bash
   python tts_app.py
   ```

## 📁 Estructura del Proyecto

```
third/
├── tts_app.py           # Aplicación principal
├── requirements.txt     # Dependencias de Python
├── history.json         # Historial de generaciones (se crea automáticamente)
├── README.md            # Este archivo
└── assets/
    ├── voices/          # Carpeta para voces personalizadas
    │   └── {nombre}/    # Cada voz tiene su carpeta
    │       ├── reference.wav
    │       ├── info.json
    │       └── latents.json (opcional)
    └── output/          # Audios generados
```

## 🎬 Uso

### Pestaña "Estudio" - Generación de Audio

1. **Selecciona una voz** del menú desplegable
2. **Escribe o importa texto**:
   - Escribe directamente en el área de texto
   - O haz clic en "Importar Archivo" para cargar TXT, PDF, DOCX o imágenes
3. **Haz clic en "GENERAR AUDIO"**
4. El audio se guardará en `assets/output/` y se añadirá al historial

### Pestaña "Mis Voces" - Crear Voces Personalizadas

1. **Escribe un nombre** para la voz (ej: "Narrador Español")
2. **Añade una descripción** (opcional)
3. **Proporciona audio de referencia**:
   - **Grabar**: Graba 6 segundos desde tu micrófono
   - **Subir WAV**: Sube un archivo WAV existente
4. **Haz clic en "GUARDAR VOZ"**

**Consejos para mejores resultados**:
- Usa grabaciones de 6-15 segundos
- Audio limpio sin ruido de fondo
- Voz clara y natural
- Formato WAV a 22050Hz mono (ideal)

### Pestaña "Biblioteca" - Historial

- **Reproduce** cualquier audio generado anteriormente
- **Elimina** entradas individuales (también borra el archivo)
- El historial se guarda automáticamente en `history.json`

## ⚙️ Configuración Avanzada

### Cambiar idioma de salida

Por defecto el idioma es español ("es"). Para cambiarlo, modifica en `tts_app.py`:

```python
self.tts_engine.generate_audio(
    text=text,
    speaker_wav=voice["audio"],
    output_path=output_path,
    language="en",  # Cambiar aquí: es, en, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko
    ...
)
```

### Duración de grabación

Cambia la duración de grabación en `VoicesTab.__init__`:

```python
self.recorder = AudioRecorder(duration=10)  # 10 segundos en vez de 6
```

## 🐛 Solución de Problemas

### "No module named 'TTS'"
```bash
pip install TTS
```

### "Tesseract not found" (OCR no funciona)
- Verifica que Tesseract está instalado: `tesseract --version`
- En macOS: `brew install tesseract`

### "MPS not available" en Mac M1/M2
- Asegúrate de tener PyTorch compatible con MPS:
```bash
pip install --upgrade torch torchaudio
```

### Audio entrecortado o con errores
- Verifica que tienes suficiente RAM (mínimo 8GB)
- Cierra otras aplicaciones pesadas
- Usa textos más cortos (divide en párrafos)

### Interfaz se ve mal en Windows
```bash
pip install --upgrade customtkinter
```

## 📝 Dependencias Principales

| Paquete | Uso |
|---------|-----|
| customtkinter | Interfaz gráfica moderna |
| TTS | Motor XTTS v2 de Coqui |
| torch | Backend de ML |
| sounddevice | Grabación de audio |
| soundfile | Lectura/escritura de audio |
| PyMuPDF | Extracción de texto de PDFs |
| python-docx | Extracción de texto de Word |
| pytesseract | OCR para imágenes |
| Pillow | Procesamiento de imágenes |

## 🔧 Hardware Recomendado

- **Mínimo**: 8GB RAM, CPU multinúcleo
- **Recomendado**: 16GB RAM, GPU con CUDA (NVIDIA) o Mac con chip M1/M2
- **Almacenamiento**: ~3GB para el modelo + espacio para audios

## 📄 Licencia

Este proyecto usa el modelo XTTS v2 de Coqui TTS, que tiene su propia licencia. Consulta https://github.com/coqui-ai/TTS para más detalles.

---

**Desarrollado con ❤️ usando Python, CustomTkinter y Coqui TTS**
