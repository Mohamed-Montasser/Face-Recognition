import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv
import pandas as pd
import numpy as np
import threading


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LBPH Face Recognition GUI")

        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.load_button = tk.Button(self.frame, text="Load ID Names", command=self.load_id_names)
        self.load_button.pack(pady=5)

        self.start_button = tk.Button(self.frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.frame, text="Stop Recognition", command=self.stop_recognition,
                                     state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.known_names_label = tk.Label(self.frame, text="Known Faces:")
        self.known_names_label.pack(pady=5)

        self.known_names_listbox = tk.Listbox(self.frame, height=10, width=30)
        self.known_names_listbox.pack(pady=5)

        self.counter_label = tk.Label(self.frame, text="Recognized: 0 / Remaining: 0")
        self.counter_label.pack(pady=5)

        self.accuracy_label = tk.Label(self.frame, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=5)

        self.running = False
        self.id_names = None
        self.recognized_faces = set()

        self.faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')
        self.lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
        self.lbph.read('Classifiers/TrainedLBPH.yml')

    def load_id_names(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.id_names = pd.read_csv(file_path)[['id', 'name']]
            self.known_names_listbox.delete(0, tk.END)
            for name in self.id_names['name']:
                self.known_names_listbox.insert(tk.END, name)
            self.update_counter()
            messagebox.showinfo("Success", "ID names loaded successfully!")

    def start_recognition(self):
        if self.id_names is None:
            messagebox.showerror("Error", "Please load ID names first!")
            return

        self.running = True
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.recognized_faces.clear()
        self.update_counter()
        threading.Thread(target=self.recognition_loop, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

    def recognition_loop(self):
        camera = cv.VideoCapture(0)
        while self.running:
            _, img = camera.read()
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = self.faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

            for x, y, w, h in faces:
                faceRegion = grey[y:y + h, x:x + w]
                faceRegion = cv.resize(faceRegion, (220, 220))
                label, trust = self.lbph.predict(faceRegion)
                try:
                    name = self.id_names[self.id_names['id'] == label]['name'].item()
                    self.recognized_faces.add(name)
                    accuracy = max(0, min(100, 100 - trust))  # Convert trust to percentage
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, f"{name} ({accuracy:.2f}%)", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                               2)
                    self.highlight_recognized_name(name)
                    self.update_accuracy(accuracy)
                except:
                    pass

            self.update_counter()
            cv.imshow('Recognize', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv.destroyAllWindows()

    def highlight_recognized_name(self, name):
        for index in range(self.known_names_listbox.size()):
            if self.known_names_listbox.get(index) == name:
                self.known_names_listbox.itemconfig(index, {'bg': 'lightgreen'})

    def update_counter(self):
        if self.id_names is not None:
            total_faces = len(self.id_names)
            recognized_count = len(self.recognized_faces)
            remaining_count = total_faces - recognized_count
            self.counter_label.config(text=f"Recognized: {recognized_count} / Remaining: {remaining_count}")

    def update_accuracy(self, accuracy):
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
