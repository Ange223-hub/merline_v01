# modules/tracker_module.py
from ultralytics import YOLO
import cv2
import numpy as np

class TrackerModule:
    """
    Module simple pour détecter des objets (personnes notamment) avec YOLOv8
    et fournir une image annotée + la liste des labels détectés.
    """

    def __init__(self, model_name="yolov8n.pt"):
        # charge le modèle YOLO. Si tu veux une autre taille, change model_name.
        self.model = YOLO(model_name)

    def detect_and_track(self, frame):
        """
        Retourne (annotated_frame, labels)
        annotated_frame : image (BGR) annotée (box + labels) en sortie de model.plot()
        labels : liste des labels détectés (chaînes, sans doublons)
        """
        labels = []
        try:
            results = self.model(frame)  # inference
            if len(results) == 0:
                return frame, labels

            r = results[0]
            # annotated image (utilise la fonction plot fournie par ultralytics)
            try:
                annotated = r.plot()  # renvoie une copie annotée
            except Exception:
                annotated = frame.copy()

            # Extraire labels de façon robuste
            try:
                # results[0].boxes.cls peut être tensor/np array
                cls_arr = getattr(r.boxes, "cls", None)
                if cls_arr is not None:
                    # convertir en liste d'ids
                    try:
                        cls_list = [int(x) for x in cls_arr]
                    except Exception:
                        # parfois boxes est un tableau numpy
                        cls_list = []
                        for box in r.boxes:
                            try:
                                cls_list.append(int(box.cls))
                            except Exception:
                                pass
                    for cid in cls_list:
                        name = self.model.names[cid] if cid in self.model.names else str(cid)
                        if name not in labels:
                            labels.append(name)
            except Exception:
                # fallback: essayer d'extraire via r.names si possible
                pass

            return annotated, labels
        except Exception as e:
            print("[TrackerModule] Erreur detection:", e)
            return frame, labels
