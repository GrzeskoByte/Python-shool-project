import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import json
import numpy as np
import os
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
        
         # Tworzymy ramkę na listę plików
        self.files_frame = tk.LabelFrame(self.root, text="Pliki")
        self.files_frame.pack(padx=10, pady=10, side="left")

        # Tworzymy listę plików
        self.files_listbox = tk.Listbox(self.files_frame)
        self.files_listbox.pack(side="left")

        # Wczytujemy pliki z folderu "files" do listy
        self.load_files()


    def load_files(self):
        # Pobieramy listę plików z folderu "files"
        files = os.listdir("files")
        # Dla każdego pliku dodajemy jego nazwę do listy
        for file in files:
            self.files_listbox.insert(tk.END, file)
            
        # Dla widgetu `self.files_listbox` wiążemy zdarzenie `<Double-Button-1>`
        # (kliknięcie lewym przyciskiem myszy podwójnie) z funkcją switchImage
        self.files_listbox.bind("<Double-Button-1>", self.switchImage)
        
    def switchImage(self, event):
        # Pobieramy indeks elementu, na którym kliknięto
        index = self.files_listbox.curselection()[0]
        # Pobieramy nazwę pliku na podstawie indeksu
        file = self.files_listbox.get(index)
        path = "./files/"+file

        with open(path, "r") as f:
            data = json.load(f)
    
        self.file_path = data["objects"][0]["file_path"]
        
        # Wczytujemy obrazek z podanej ścieżki
        image = Image.open(self.file_path)
        image = image.resize((300, 300))
        # Konwertujemy obrazek na format zgodny z tkinter
        image = ImageTk.PhotoImage(image)
        # Zmieniamy obrazek wyświetlany w widget self.image_label na nowy obrazek
        self.image_label.configure(image=image)
        self.image_label.image = image
       
   
      
       


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
            # Pobierz współrzędne obiektu i nazwę
            if(obj["type"]=="truck"):
                x = obj["x"]
                y = obj["y"]
                w = obj["width"]
                h = obj["height"]
                name = obj["type"]

                # Zdefiniuj punkty współrzędnych obrysu obiektu
                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                # Narysuj obrysy obiektu na obrazie
                cv2.polylines(image, [points], True, (0, 255, 0), 2)
                # Wypisz nazwę obiektu na obrazie
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

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

        file_name, file_extension = os.path.splitext(self.file_path)
        # Odseparowujemy katalogi od nazwy pliku
        _, file_name = os.path.split(file_name)
        name="./files/"+ file_name + ".json"
       
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            data["objects"].append({
                "type": class_names[class_ids[i]],
                "file_path":self.file_path,
                "file_name":file_name,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })
        
     
        
        # Zapisz dane do pliku JSON
        with open(name, "w") as f:
             json.dump(data, f)

        self.json_path = name
        self.draw_objects_from_json()

   
root = tk.Tk()
app = App(root)
root.mainloop()
