# modules/voice_module.py
import threading
import pyttsx3

# Optional: speech recognition is supported if installed, but we keep it separate
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

class VoiceModule:
    def __init__(self, lang="fr-FR", rate=160, volume=1.0, callback=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self.lang = lang
        self.callback = callback  # optional function(text) called when we got speech
        self._running = False

        # speech recognition attributes (optional)
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None

    def speak(self, text):
        """Speak synchronously (blocking)."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("[VoiceModule] Erreur TTS:", e)

    def _listen_loop(self):
        """Background listening loop (uses speech_recognition if available)."""
        if not SR_AVAILABLE:
            print("[VoiceModule] speech_recognition non disponible.")
            return

        mic = sr.Microphone()
        while self._running:
            try:
                with mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=5)
                txt = self.recognizer.recognize_google(audio, language=self.lang)
                print("[VoiceModule] Heard:", txt)
                if self.callback:
                    try:
                        self.callback(txt)
                    except Exception as e:
                        print("[VoiceModule] callback error:", e)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print("[VoiceModule] RequestError:", e)
                continue
            except Exception as e:
                print("[VoiceModule] Unexpected listen error:", e)
                continue

    def start_listening(self):
        """Start background listening thread (if SR available)."""
        if not SR_AVAILABLE:
            print("[VoiceModule] speech_recognition not installed, cannot start listening.")
            return
        if not self._running:
            self._running = True
            t = threading.Thread(target=self._listen_loop, daemon=True)
            t.start()

    def stop_listening(self):
        self._running = False
