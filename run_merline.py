# run_merline.py
import cv2
from modules.face_module import FaceModule
from modules.tracker_module import TrackerModule
from modules.voice_module import VoiceModule
from datetime import datetime
import os

# create captures dir if needed
os.makedirs("captures", exist_ok=True)
os.makedirs("captures/stephane", exist_ok=True)

# --- Initialisation des modules ---
face_module = FaceModule()               # detection + embeddings
tracker_module = TrackerModule()         # YOLO detection
voice_module = VoiceModule()             # TTS (et optionally SR)

# --- Initialisation webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    exit()

# --- Variable pour stocker les visages déjà salués ---
saluted_faces = set()

print("Merline démarrée. Appuie sur 's' pour enregistrer une image, 'q' pour quitter.")

# --- Boucle principale ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : image non récupérée.")
        break

    # --- Détection + tracking (YOLO) ---
    annotated_frame, labels = tracker_module.detect_and_track(frame)

    # --- Détection des visages (InsightFace) ---
    locations, names = face_module.detect_faces(frame)

    # Dessiner les bboxes faciales et noms sur l'image annotée
    for bbox, name in zip(locations, names):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if name == "Stéphane" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Salutation (une fois par nom)
        if name not in saluted_faces:
            greeting = face_module.get_greeting()
            if name == "Stéphane":
                voice_module.speak(f"{greeting} Stéphane ! hey j'te vois !")
            else:
                voice_module.speak("Bonjour, je ne te connais pas. Peux-tu te présenter ?")
            saluted_faces.add(name)

    # --- Affichage message YOLO ---
    if "person" in labels:
        message = f"{face_module.get_greeting()} Personne détectée"
        color = (0, 255, 0)
    else:
        message = "Je ne vois personne..."
        color = (0, 0, 255)

    cv2.putText(annotated_frame, message, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- Affichage vidéo ---
    cv2.imshow("MERLINE Vision", annotated_frame)

    # --- Gestion des touches ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save a capture for training
        fname = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join("captures", "stephane", fname)
        cv2.imwrite(save_path, frame)
        # reload known embeddings quickly
        face_module.load_known_embeddings()
        voice_module.speak("Image enregistrée pour l'entraînement.")
        print("[run_merline] Image sauvegardée →", save_path)
    elif key == ord('q'):
        break

# --- Libération des ressources ---
cap.release()
cv2.destroyAllWindows()
print("Fin du programme. Webcam fermée.")
