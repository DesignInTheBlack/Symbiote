import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Optional
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}[%(asctime)s]{Style.RESET_ALL} %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VoiceServiceV2")

app = FastAPI(title="Symbiote Voice Service V2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
class ServiceState:
    def __init__(self):
        self.tts_ready = False
        self.stt_ready = False
        self.vad_ready = False
        self.tts_model = None
        self.stt_model = None
        self.vad_model = None
        self.vocab = None # For STT tokenizer

STATE = ServiceState()

# --- Voice Effects Processor ---
class VoiceEffectsProcessor:
    """Applies audio effects to TTS output using pedalboard."""
    
    def __init__(self):
        self.enabled = False
        try:
            from pedalboard import Pedalboard, PitchShift, Reverb, Compressor, Gain
            self.Pedalboard = Pedalboard
            self.PitchShift = PitchShift
            self.Reverb = Reverb
            self.Compressor = Compressor
            self.Gain = Gain
            self.enabled = True
            logger.info("Pedalboard loaded for voice effects.")
        except ImportError:
            logger.warning("Pedalboard not installed. Voice effects disabled.")
    
    def apply(self, audio: np.ndarray, sample_rate: int,
              pitch_semitones: float = 0.0,
              reverb_amount: float = 0.0,
              compression: float = 0.0,
              formant_shift: float = 0.0) -> np.ndarray:
        """
        Apply voice effects to audio.
        
        Args:
            audio: Audio as float32 numpy array (-1 to 1)
            sample_rate: Sample rate (typically 24000 for Kokoro)
            pitch_semitones: Pitch shift in semitones (-12 to +12)
            reverb_amount: Reverb room size (0.0 to 1.0)
            compression: Compression amount (0.0 to 1.0)
            formant_shift: Formant shift (not implemented yet, placeholder)
        
        Returns:
            Processed audio as float32 numpy array
        """
        if not self.enabled:
            return audio
        
        # Skip if no effects requested
        if pitch_semitones == 0.0 and reverb_amount == 0.0 and compression == 0.0:
            return audio
        
        effects = []
        
        # Pitch shift
        if pitch_semitones != 0.0:
            effects.append(self.PitchShift(semitones=pitch_semitones))
        
        # Reverb
        if reverb_amount > 0.0:
            effects.append(self.Reverb(
                room_size=min(reverb_amount, 1.0),
                wet_level=reverb_amount * 0.5,
                dry_level=1.0 - (reverb_amount * 0.3)
            ))
        
        # Compression
        if compression > 0.0:
            # Map 0-1 to reasonable threshold range (-40dB to -10dB)
            threshold_db = -40 + (compression * 30)
            effects.append(self.Compressor(
                threshold_db=threshold_db,
                ratio=2.0 + (compression * 4),  # 2:1 to 6:1
                attack_ms=5.0,
                release_ms=100.0
            ))
            # Add makeup gain
            effects.append(self.Gain(gain_db=compression * 6))
        
        if not effects:
            return audio
        
        board = self.Pedalboard(effects)
        
        # Ensure audio is 2D for pedalboard (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        processed = board(audio, sample_rate)
        
        # Return to 1D if input was 1D
        if processed.ndim == 2 and processed.shape[0] == 1:
            processed = processed.flatten()
        
        return processed

EFFECTS_PROCESSOR = VoiceEffectsProcessor()

# --- Config ---
# --- Config ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Wrappers ---

class TenVadWrapper:
    def __init__(self):
        try:
            from ten_vad import TenVad
            # Initialize with defaults (hop_size=256, threshold=0.5)
            self.detector = TenVad(hop_size=256, threshold=0.5) 
            self.buffer = b""
            logger.info("TenVad initialized.")
        except Exception as e:
            logger.error(f"Failed to init TenVad: {e}. Running in mock mode.")
            self.detector = None

    def process(self, audio_chunk: bytes) -> bool:
        """Returns True if speech is detected in the chunk."""
        if not self.detector:
            return False
        
        # Buffer input
        self.buffer += audio_chunk
        
        # Process in chunks of 512 bytes (256 samples * 2 bytes/sample)
        # 256 samples @ 16kHz = 16ms (Matches TenVad default hop_size)
        chunk_size = 512
        is_speech = False
        
        while len(self.buffer) >= chunk_size:
            frame_bytes = self.buffer[:chunk_size]
            self.buffer = self.buffer[chunk_size:]
            
            # Convert to numpy int16 (TenVad expects int16)
            audio_data = np.frombuffer(frame_bytes, dtype=np.int16)
            
            # Debug: Check amplitude occasionally
            if np.random.rand() < 0.005:
               max_amp = np.max(np.abs(audio_data))
               logger.info(f"Audio Max Amp: {max_amp}")

            try:
                # Returns (probability, state) based on discovery script
                # Type: <class 'tuple'>
                prob, _ = self.detector.process(audio_data)
                
                # Debug VAD output
                # if prob > 0.1: 
                #    logger.info(f"VAD p: {prob:.4f}")

                if prob > 0.5:
                    is_speech = True
                    # logger.info(f"VAD Trigger: {prob}")
                        
            except Exception as e:
                logger.error(f"VAD Error: {e}")
            
        return is_speech

class ParakeetWrapper:
    def __init__(self):
        self.session = None
        self.processor = None
        self.tokenizer = None
        
    def load(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        
        repo_id = "istupakov/parakeet-ctc-0.6b-onnx"
        
        logger.info(f"Checking for Parakeet model in {MODELS_DIR}...")
        
        # 1. Download/Cache Model
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx", cache_dir=MODELS_DIR)
        
        # External data
        try:
             hf_hub_download(repo_id=repo_id, filename="model.onnx.data", cache_dir=MODELS_DIR)
        except Exception:
             pass

        # 2. Load Vocab
        try:
            vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.txt", cache_dir=MODELS_DIR)
            self.load_vocab(vocab_path)
        except Exception as e:
            logger.error(f"Failed to load vocab: {e}")

        # 3. Init ONNX Session (GPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        logger.info(f"Parakeet ONNX loaded on {self.session.get_providers()[0]}")

    def load_vocab(self, path):
        self.id2token = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token = parts[0]
                    idx = int(parts[-1])
                    self.id2token[idx] = token
        
        # Determine blank token (usually last or 0? Parakeet usually last)
        # Search says <blk> often last.
        # But let's check if there's a specific blank.
        # Often len(vocab) is blank index for ONNX models if implicit?
        # Or <unk> is 0.
        # Let's assume standard CTC: blank is often 0 or len(vocab)-1.
        # For Nemo Parakeet, blank is usually the last index = len(vocab).
        # We will use len(self.id2token) as blank if not found.
        self.blank_id = len(self.id2token) # Approximate? 
        # Actually safer to rely on argmax values. If argmax == len(vocab), it's blank.
    
    def decode(self, token_ids):
        # CTC Greedy Decode with proper blank handling
        
        if not self.id2token:
            return "Error: Vocab not loaded"
        
        # Step 1: Collapse consecutive duplicates (CTC rule)
        collapsed_ids = []
        prev_id = None
        for tid in token_ids:
            if tid != prev_id:
                collapsed_ids.append(tid)
                prev_id = tid
        
        # Step 2: Remove blanks and convert to tokens
        # CTC blank is typically the last index (len(vocab))
        # or explicitly marked as <blk>
        blank_id = len(self.id2token)  # Standard CTC blank index
        
        decoded_tokens = []
        for tid in collapsed_ids:
            # Skip CTC blank (either by ID or by token name)
            if tid == blank_id or tid not in self.id2token:
                continue
            
            token = self.id2token[tid]
            
            # Skip special tokens
            if token.startswith("<") and token.endswith(">"):
                continue
            
            decoded_tokens.append(token)
        
        # Step 3: Join and handle SentencePiece formatting
        # ▁ (U+2581) represents word boundaries in SentencePiece
        text = "".join(decoded_tokens)
        
        # Replace ▁ with spaces and clean up
        text = text.replace("▁", " ").strip()
        
        # Clean up multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def compute_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute Log Mel Spectrogram matching NeMo/Parakeet defaults.
        SR=16k, n_mels=80, n_fft=512, hop=160 (10ms), win=400 (25ms).
        """
        import librosa
        # NeMo defaults
        SAMPLE_RATE = 16000
        N_FFT = 512
        HOP_LENGTH = 160
        WIN_LENGTH = 400
        N_MELS = 80
        
        mel = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            win_length=WIN_LENGTH, 
            n_mels=N_MELS,
            center=True,
            pad_mode="reflect"
        )
        
        # Log Mel: log(x + 1e-5)
        log_mel = np.log(mel + 1e-5)
        
        # Normalization (Per-Instance Zero-Mean Unit-Variance)
        # This is standard for Conformer/NeMo models if 'dither' or 'pad' isn't huge.
        mean = np.mean(log_mel, axis=1, keepdims=True)
        std = np.std(log_mel, axis=1, keepdims=True)
        log_mel_norm = (log_mel - mean) / (std + 1e-5)
        
        # Add Batch dimension: [1, 80, T]
        return log_mel_norm[np.newaxis, :, :].astype(np.float32)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribes raw 16kHz PCM audio."""
        if not self.session:
            return ""
        
        # Convert bytes to float32 array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        try:
            # 1. Compute Features (Log Mel)
            features = self.compute_features(audio_data)
            
            # 2. Compute Length
            # Shape is [1, 80, T]
            T = features.shape[2]
            audio_len = np.array([T], dtype=np.int64)
            
            # 3. Build Inputs
            inputs = {}
            for inp in self.session.get_inputs():
                if 'audio_signal' in inp.name:
                    inputs[inp.name] = features
                elif 'length' in inp.name:
                    inputs[inp.name] = audio_len
            
            if not inputs:
                 logger.error("Could not map inputs.")
                 return ""

            # 4. Run Inference
            logits = self.session.run(None, inputs)[0]
            
            # 5. Greedy Decode
            token_ids = np.argmax(logits, axis=-1)[0]
            text = self.decode(token_ids)
            return text
            
        except Exception as e:
            logger.error(f"Inference failed (Feature Extraction?): {e}")
            import traceback
            # traceback.print_exc()
            return ""

# --- Loaders ---

def load_vad_model():
    global STATE
    logger.info("Loading TEN-VAD...")
    try:
        STATE.vad_model = TenVadWrapper()
        STATE.vad_ready = True
        logger.info(f"{Fore.GREEN}TEN-VAD Wrapper Ready.")
    except Exception as e:
        logger.error(f"Failed to init TEN-VAD: {e}")

def load_stt_model():
    global STATE
    logger.info("Loading Parakeet STT...")
    try:
        wrapper = ParakeetWrapper()
        wrapper.load()
        STATE.stt_model = wrapper
        STATE.stt_ready = True
        logger.info(f"{Fore.GREEN}Parakeet STT Ready.")
    except Exception as e:
        logger.error(f"Failed to load Parakeet: {e}")
        import traceback
        traceback.print_exc()

def load_tts_model():
    global STATE
    logger.info("Loading Kokoro TTS...")
    try:
        from kokoro import KPipeline
        # Init pipeline for American English
        # Force CPU/GPU check? KPipeline handles it.
        pipeline = KPipeline(lang_code='a') 
        STATE.tts_model = pipeline
        STATE.tts_ready = True
        logger.info(f"{Fore.GREEN}Kokoro TTS Loaded.")
    except Exception as e:
        logger.error(f"Failed to load Kokoro TTS: {e}")

# --- Routes ---

@app.on_event("startup")
async def startup_event():
    # Helper thread to load models without blocking server boot
    t = threading.Thread(target=background_loader, daemon=True)
    t.start()

@app.get("/health")
def health_check():
    """Returns the ready state of all components."""
    return {
        "status": "ready" if (STATE.tts_ready and STATE.stt_ready and STATE.vad_ready) else "loading",
        "components": {
            "tts": STATE.tts_ready,
            "stt": STATE.stt_ready,
            "vad": STATE.vad_ready
        }
    }

@app.post("/tts")
def tts_generate(text: str):
    """Simple HTTP TTS endpoint (Non-streaming fallback)."""
    if not STATE.tts_ready:
        return {"error": "TTS not ready"}
    return {"message": "Use WebSocket for TTS"}

# --- Background Loader ---
def background_loader():
    logger.info("Starting background model loading...")
    load_vad_model()
    load_tts_model()
    load_stt_model()
    
    if STATE.vad_ready and STATE.tts_ready and STATE.stt_ready:
        logger.info(f"{Fore.GREEN}{Style.BRIGHT}Voice Service V2 Fully Ready!")
    else:
        logger.warning(f"Voice Service started with partial failures. VAD:{STATE.vad_ready} STT:{STATE.stt_ready} TTS:{STATE.tts_ready}")

# --- WebSocket: Audio Stream & Events ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    
    # Session State
    audio_buffer = bytearray()
    is_speaking = False
    silence_frames = 0
    
    try:
        while True:
            try:
                # We handle text and binary together
                # receive() returns a dict with 'type', 'bytes', or 'text'
                message = await websocket.receive()
            except RuntimeError as e:
                # This happens if Starlette already received a disconnect
                if "disconnect" in str(e).lower():
                    logger.info("WebSocket disconnected (RuntimeError catch)")
                    break
                raise e
            
            if "bytes" in message:
                chunk = message["bytes"]
                
                # 1. VAD Check
                speech_detected = False
                if STATE.vad_ready and STATE.vad_model:
                     speech_detected = STATE.vad_model.process(chunk)
                
                audio_buffer.extend(chunk)
                
                # 2. Logic: If speech starts -> Interrupt
                if speech_detected:
                    silence_frames = 0
                    if not is_speaking:
                        is_speaking = True
                        logger.info("Speech detected! Interrupting...")
                        await websocket.send_json({"type": "interrupt"})
                        # Keep recent buffer for context? For now, we just keep accumulating.
                
                # 3. Logic: If silence -> Transcribe
                if not speech_detected and is_speaking:
                   silence_frames += 1
                   
                   # 16kHz audio, 512 byte chunks (256 samples) ~ 16ms
                   # 30 frames ~ 480ms silence
                   if silence_frames > 30: 
                       logger.info("Silence detected. Transcribing...")
                       if STATE.stt_ready and STATE.stt_model:
                           try:
                               text = STATE.stt_model.transcribe(bytes(audio_buffer))
                               if text and text.strip():
                                   logger.info("Transcribed (redacted). Length=%d", len(text.strip()))
                                   await websocket.send_json({"type": "text", "content": text})
                           except Exception as e:
                               logger.error(f"Transcription failed: {e}")
                       
                       audio_buffer = bytearray()
                       is_speaking = False
                       silence_frames = 0
                pass
            
            if "text" in message:
                data = json.loads(message["text"])
                
                # Handle stop listening command
                if data.get("type") == "stop_listening":
                    logger.info("Stop listening command received - resetting VAD state")
                    audio_buffer = bytearray()
                    is_speaking = False
                    silence_frames = 0
                    continue
                
                # Handle TTS request
                if data.get("type") == "tts":
                     text = data.get("content")
                     voice = data.get("voice", "af_bella")  # Default to af_bella
                     speed = data.get("speed", 1.0)  # Default to 1.0
                     
                     # Voice effects parameters
                     pitch_semitones = data.get("pitch_semitones", 0.0)
                     reverb_amount = data.get("reverb_amount", 0.0)
                     compression = data.get("compression", 0.0)
                     formant_shift = data.get("formant_shift", 0.0)
                     
                     try:
                         if not STATE.tts_model:
                             logger.error("TTS model not ready")
                             await websocket.send_json({"type": "tts_error", "detail": "tts_model_not_ready"})
                         else:
                             # KPipeline returns: (graphemes, phonemes, audio_float)
                             # We iterate generator
                             stream = STATE.tts_model(text, voice=voice, speed=speed, split_pattern=r'\n+')
                             for _, _, audio in stream:
                                 # Check connection state
                                 from fastapi.websockets import WebSocketState
                                 if websocket.client_state == WebSocketState.DISCONNECTED:
                                     logger.info("TTS Loop Broken: Client Disconnected")
                                     break

                                 # Convert PyTorch Tensor -> NumPy
                                 # Kokoro returns torch tensors, need to convert to numpy first
                                 audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else audio
                                 
                                 # Apply voice effects (if any configured)
                                 audio_np = EFFECTS_PROCESSOR.apply(
                                     audio_np.astype(np.float32),
                                     sample_rate=24000,  # Kokoro output rate
                                     pitch_semitones=pitch_semitones,
                                     reverb_amount=reverb_amount,
                                     compression=compression,
                                     formant_shift=formant_shift
                                 )
                                 
                                 # Convert to Int16 PCM
                                 pcm = (audio_np * 32767).astype(np.int16).tobytes()
                                 await websocket.send_bytes(pcm)
                     except Exception as e:
                         logger.error(f"TTS Error: {e}")
                         try:
                             await websocket.send_json({"type": "tts_error", "detail": str(e)})
                         except Exception:
                             pass
                     finally:
                         try:
                             await websocket.send_json({"type": "tts_end"})
                         except Exception:
                             pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run on localhost:11435 (Distinct from Ollama 11434)
    uvicorn.run(app, host="127.0.0.1", port=11435)
