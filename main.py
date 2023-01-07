import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import json
import numpy as np
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Zdjęcia")

        # Tworzymy przycisk do wczytywania zdjęcia
        self.load_button = tk.Button(self.root, text="Wczytaj zdjęcie", command=self.load_image)
        self.label_button = tk.Button(self.root, text="Etykietuj", command=self.detect_objects)
        self.load_button.pack(padx=10, pady=10)
        self.label_button.pack(padx=10, pady=12)

        # Tworzymy pole z obrazkiem
        self.image_frame = tk.LabelFrame(self.root, text="Wyświetlone zdjęcie")
        self.image_frame.pack(padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

    def load_image(self):
        # Otwieramy okno dialogowe do wyboru pliku
        file_path = filedialog.askopenfilename()
        self.file_path = file_path
        # Jeśli użytkownik wybrał plik
        if file_path:
            # Próbujemy otworzyć plik jako obraz
            try:
                image = Image.open(file_path)
                image = image.resize((300, 300))
                image = ImageTk.PhotoImage(image)
                self.image_label.configure(image=image)
                self.image_label.image = image
            except:
                # Jeśli otwarcie pliku jako obrazu się nie powiedzie, wyświetlamy komunikat o błędzie
                messagebox.showerror("Błąd", "Nie można otworzyć pliku jako obrazu")
            
    def draw_objects(self):
        # Wczytaj zdjęcie i przekształć je na skalę szarości
        image = cv2.imread(self.file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Znajdź kontury obiektów
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Dla każdego konturu
        for contour in contours:
            # Narysuj kontur na obrazie
            cv2.drawContours(image, contour, -1, (0, 255, 0), 2)

        # Wyświetl obraz z obrysem obiektów
        resized_image = cv2.resize(image, (600, 600))
        cv2.imshow("Obiekty", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_objects_from_json(self):
        # Wczytaj zdjęcie i przekształć je na skalę szarości
        image = cv2.imread(self.file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Wczytaj dane z pliku JSON
        with open(self.json_path, "r") as f:
            data = json.load(f)

        # Dla każdego obiektu zapisanego w słowniku "data"
        for obj in data["objects"]:
            # Pobierz współrzędne obiektu
            if(obj["type"]=="car"):
                x = obj["x"]
                y = obj["y"]
                w = obj["width"]
                h = obj["height"]

                # Zdefiniuj punkty współrzędnych obrysu obiektu
                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                # Narysuj obrysy obiektu na obrazie
                cv2.polylines(image, [points], True, (0, 255, 0), 2)

        # Wyświetl obraz z obrysem obiektów
        resized_image = cv2.resize(image, (600, 600))
        cv2.imshow("Obiekty", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def detect_objects(self):
    # Wczytaj zdjęcie
        image = cv2.imread(self.file_path)

        # Pobierz model YOLO z repozytorium OpenCV
        model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

        # Przygotuj model do użycia
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Przygotuj zdjęcie do przetworzenia przez model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Wykonaj przez model predykcję
        model.setInput(blob)
        output_layers = model.getUnconnectedOutLayersNames()
        output = model.forward(output_layers)

        # Przetwórz wynik predykcji
        boxes = []
        confidences = []
        class_ids = []
        for output_layer in output:
            for detection in output_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    center_x, center_y, width, height = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Zastosuj filtrowanie pudełek
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Przygotuj słownik z współrzędnymi obiektów
        data = {
            "objects": []
        }
        # Przygotuj zdjęcie do przetworzenia przez model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Wykonaj przez model predykcję
        model.setInput(blob)
        output_layers = model.getUnconnectedOutLayersNames()
        output = model.forward(output_layers)

        # Przetwórz wynik predykcji
        boxes = []
        confidences = []
        class_ids = []
        for output_layer in output:
            for detection in output_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    center_x, center_y, width, height = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Zastosuj filtrowanie pudełek
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Przygotuj słownik z współrzędnymi obiektów
        data = {
            "objects": []
        }

        # Wczytaj nazwy klas z pliku "coco.names"
        with open("coco.names", "r") as f:
            class_names = f.read().strip().split("\n")

        # Dla każdego znalezionego obiektu dodaj do słownika jego współrzędne oraz typ
        
       
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            data["objects"].append({
                "type": class_names[class_ids[i]],
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })



        # Zapisz dane do pliku JSON
        with open("files/objects.json", "w") as f:
            json.dump(data, f)
        self.json_path = "./files/objects.json"
        self.draw_objects_from_json()

   


root = tk.Tk()
app = App(root)
root.mainloop()
