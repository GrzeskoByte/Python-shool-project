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

       
        self.load_button = tk.Button(self.root, text="Wczytaj zdjęcie", command=self.load_image)
        self.label_button = tk.Button(self.root, text="Etykietuj", command=self.detect_objects)
        self.load_button.pack(padx=10, pady=10)
        self.label_button.pack(padx=10, pady=12)

        
        self.image_frame = tk.LabelFrame(self.root, text="Wyświetlone zdjęcie")
        self.image_frame.pack(padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        
        self.files_frame = tk.LabelFrame(self.root, text="Pliki")

        scrollbar = tk.Scrollbar(self.files_frame)
        scrollbar.pack(side="right", fill="y")

        self.files_listbox = tk.Listbox(self.files_frame, yscrollcommand=scrollbar.set)
        self.files_listbox.pack()
  
        scrollbar.config(command=self.files_listbox.yview)

        self.files_frame.pack(padx=10, pady=10, side="left")


        
        self.load_files()


    def load_files(self):
        
        files = os.listdir("files")
        
        for file in files:
            self.files_listbox.insert(tk.END, file)
            
        self.files_listbox.bind("<Double-Button-1>", self.switchImage)
        
   

    def switchImage(self, event):
        
        index = self.files_listbox.curselection()[0]
       
        file = self.files_listbox.get(index)
        path = "./files/"+file

        with open(path, "r") as f:
            data = json.load(f)
    
        self.file_path = data["objects"][0]["file_path"]
        
        
        image = Image.open(self.file_path)
        image = image.resize((300, 300))
        
        image = ImageTk.PhotoImage(image)
        
        self.image_label.configure(image=image)
        self.image_label.image = image
       

    def load_image(self):
        file_paths = filedialog.askopenfilenames()
        self.file_paths = file_paths

        for file_path in file_paths:
            try:
                image = Image.open(file_path)
                image = image.resize((300, 300))
                image = ImageTk.PhotoImage(image)
                self.image_label.configure(image=image)
                self.image_label.image = image

              
                object_data = {"objects": [{"file_path": file_path, "objects_detected": []}]}
                filename = file_path.split("/")[-1].split(".")[0]
                json_file = f"files/{filename}.json"
                with open(json_file, "w") as f:
                    json.dump(object_data, f)
                    
                self.load_files()
            except:
                
                messagebox.showerror("Błąd", "Nie można otworzyć pliku jako obrazu")
            
    def draw_objects_from_json(self):
        
        image = cv2.imread(self.file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      
        with open(self.json_path, "r") as f:
            data = json.load(f)

       
        for obj in data["objects"]:
            
            if(obj["label"]=="truck"):
                x = obj["x"]
                y = obj["y"]
                w = obj["width"]
                h = obj["height"]
                name = obj["label"]

              
                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)

                cv2.polylines(image, [points], True, (0, 255, 0), 2)
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

        
        resized_image = cv2.resize(image, (600, 600))
        cv2.imshow("Obiekty", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def detect_objects(self):
   
        image = cv2.imread(self.file_path)

      
        model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

     
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

       
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

       
        model.setInput(blob)
        output_layers = model.getUnconnectedOutLayersNames()
        output = model.forward(output_layers)
        
       
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

        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        data = {
            "objects": []
        }
       
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        
        model.setInput(blob)
        output_layers = model.getUnconnectedOutLayersNames()
        output = model.forward(output_layers)

       
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

       
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        
        data = {
            "objects": []
        }

        with open("coco.names", "r") as f:
            class_names = f.read().strip().split("\n")


        file_name, file_extension = os.path.splitext(self.file_path)
       
        _, file_name = os.path.split(file_name)
        name="./files/"+ file_name + ".json"
        
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            data["objects"].append({
                "label": class_names[class_ids[i]],
                "file_path":self.file_path,
                "file_name":file_name,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
             
            })
        
       
        with open(name, "w") as f:
             json.dump(data, f)

        self.json_path = name
        self.draw_objects_from_json()
       

         
root = tk.Tk()
app = App(root)
root.mainloop()
