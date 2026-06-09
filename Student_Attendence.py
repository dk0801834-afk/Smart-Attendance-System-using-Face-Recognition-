# Student Attendence System using Face Recognition



# Here We use different libraries for different purposes:
"""
1We use OpenCV for face detection and recognition.
2 We use os for handling file and foler.
3 We use csv for handling csv files.
4 We use pandas for data manipulation.
5 We use datetime for handling date and time.
6 We use tkinter for creating GUI.

"""
import cv2  # type: ignore[import]
import os
import csv
import pandas as pd # type: ignore[import]
from datetime import datetime
import tkinter as tk
from tkinter import messagebox


# Here We are creating folders for storing and training
if not os.path.exists("images"):
    os.makedirs("images")

if not os.path.exists("trainer"):
    os.makedirs("trainer")

if not os.path.exists("attendance"):
    os.makedirs("attendance")


student_file = "students.csv"

if not os.path.exists(student_file):
    with open(student_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name"])
        
        
# We are creating this function for student registration and storing the data in csv file.
def register_student():
    sid = entry_id.get()
    name = entry_name.get()

    if sid == "" or name == "":
        messagebox.showerror("Error", "Enter ID and Name")
        return

    with open(student_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([sid, name])

    messagebox.showinfo("Success", "Student Registered")


# We are creating this function for capturing face images and storing them in images folder.
def capture_face():
    sid = entry_id.get()

    if sid == "":
        messagebox.showerror("Error", "Enter Student ID")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]

            cv2.imwrite(f"images/User.{sid}.{count}.jpg", face)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Capture Face", img)

        if cv2.waitKey(1) == ord("q") or count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Success", "Face Images Captured")
    
# We are creating this function for training the face images and storing the trained data in trainer folder.
def train_data():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    image_paths = [os.path.join("images", f) for f in os.listdir("images")]

    faces = []
    ids = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        sid = int(os.path.split(path)[-1].split(".")[1])

        faces.append(img)
        ids.append(sid)

    recognizer.train(faces, pd.Series(ids).values)
    recognizer.save("trainer/trainer.yml")

    messagebox.showinfo("Success", "Training Complete")


# We are creating this function for marking attendance using face recognition and storing the attendance in attendance folder.
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    df = pd.read_csv(student_file)

    cam = cv2.VideoCapture(0)

    marked = []

    today = datetime.now().strftime("%d-%m-%Y")
    filename = f"attendance/{today}.csv"

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 60:
                name = df.loc[df["ID"] == id]["Name"].values

                if len(name) > 0:
                    name = name[0]
                else:
                    name = "Unknown"

                text = f"{id}-{name}"

                if id not in marked:
                    now = datetime.now().strftime("%H:%M:%S")

                    with open(filename, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([id, name, today, now, "Present"])

                    marked.append(id)

            else:
                text = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Attendance", img)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    
    
    
# We are creating GUI using tkinter for the attendance system.  
root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("500x450")    
    
    #
tk.Label(root, text="Student ID").pack()
entry_id = tk.Entry(root)
entry_id.pack()   
    
    
tk.Label(root, text="Student Name").pack()
entry_name = tk.Entry(root)
entry_name.pack()
   
    
    
tk.Button(root, text="Register Student", width=25,
        command=register_student).pack(pady=5)   
    
    
    
tk.Button(root, text="Capture Face", width=25,
          command=capture_face).pack(pady=5)    
    
    
    
    
tk.Button(root, text="Train Data", width=25,
        command=train_data).pack(pady=5)

    
tk.Button(root, text="Start Attendance", width=25,
          command=mark_attendance).pack(pady=5)
  
    
tk.Button(root, text="Exit", width=25,
          command=root.destroy).pack(pady=5)   
 
root.mainloop()
