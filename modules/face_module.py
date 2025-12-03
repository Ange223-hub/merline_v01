# modules/face_module.py
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from datetime import datetime

class FaceModule:
    """
    Détection + reconnaissance faciale basée sur InsightFace (SCRFD + ArcFace).
    Fournit detect_faces(frame) -> (locations, names)
    """

    def __init__(self, steph_dir="captures/stephane", det_size=(640, 640), model_name="buffalo_l"):
        self.steph_dir = steph_dir
        os.makedirs(self.steph_dir, exist_ok=True)

        # initialise InsightFace (SCRFD detection + ArcFace embeddings)
        # providers=["CPUExecutionProvider"] fonctionne avec onnxruntime CPU
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        # prepare peut télécharger et charger des poids la première fois
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.known_embeddings = []  # list of numpy arrays
        self.load_known_embeddings()

    def load_known_embeddings(self):
        """Charge les images présentes dans captures/stephane et calcule leurs embeddings."""
        self.known_embeddings = []
        try:
            files = [f for f in os.listdir(self.steph_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            for f in files:
                path = os.path.join(self.steph_dir, f)
                img = cv2.imread(path)
                if img is None:
                    continue
                faces = self.app.get(img)
                if len(faces) > 0:
                    emb = faces[0].embedding.astype(np.float32)
                    self.known_embeddings.append(emb)
        except Exception as e:
            print("[FaceModule] load_known_embeddings error:", e)
        print(f"[FaceModule] Loaded {len(self.known_embeddings)} known embeddings")

    def add_image_and_embedding(self, img_bgr, save=True):
        """Ajoute une image (BGR) pour apprendre un nouveau visage et sauvegarde l'image."""
        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            return False
        emb = faces[0].embedding.astype(np.float32)
        self.known_embeddings.append(emb)
        if save:
            fname = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            path = os.path.join(self.steph_dir, fname)
            cv2.imwrite(path, img_bgr)
        return True

    def _is_match(self, emb, threshold=0.35):
        """Cosine similarity matching against known embeddings."""
        if len(self.known_embeddings) == 0:
            return False, 0.0
        sims = []
        for k in self.known_embeddings:
            denom = (np.linalg.norm(emb) * np.linalg.norm(k))
            if denom == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(emb, k) / denom))
        best = max(sims)
        return (best >= threshold), best

    def detect_faces(self, img_bgr, threshold=0.35):
        """
        Detecte visages et renvoie (locations, names)
        locations : liste de bbox [x1,y1,x2,y2]
        names : liste de noms correspondant ("Stéphane" ou "Inconnu")
        """
        locations = []
        names = []
        try:
            faces = self.app.get(img_bgr)
            for f in faces:
                bbox = f.bbox.astype(int).tolist()
                emb = f.embedding.astype(np.float32)
                match, score = self._is_match(emb, threshold=threshold)
                if match:
                    name = "Stéphane"
                else:
                    name = "Inconnu"
                locations.append(bbox)
                names.append(name)
        except Exception as e:
            print("[FaceModule] detect_faces error:", e)
        return locations, names

    def get_greeting(self):
        """Retourne la salutation adaptée à l'heure."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Bonjour"
        elif 12 <= hour < 18:
            return "Bon après-midi"
        elif 18 <= hour < 22:
            return "Bonsoir"
        else:
            return "Bonne nuit"
