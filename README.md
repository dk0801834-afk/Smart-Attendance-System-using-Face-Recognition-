# Smart-Attendance-System-using-Face-Recognition-
A desktop application built with Python and Computer Vision that automates attendance tracking in real-time. It provides a graphical user interface (GUI) to register users, capture facial datasets, train local machine learning models, and log attendance directly into CSV files.

🚀 Features
  -->Student Registration: Easily add new students with a unique ID and Name, saving their details to a master CSV registry.

  -->Automated Face Capture: Uses OpenCV Haar Cascades to detect faces and captures up to 30 grayscale images per student for dataset creation.

  -->Local ML Training: Employs an LBPH (Local Binary Patterns Histograms) face recognizer to train on the captured image dataset and save the model instantly.

  -->Real-Time Attendance Scanning: Matches faces via webcam against the trained model, verifies IDs against the registry, and logs exact arrival timestamps           ("Present") in daily CSV logs.

  -->Interactive GUI: Built using Tkinter to provide a seamless, click-to-action dashboard.

🛠️ Tech Stack & Prerequisites
Language: Python 3.x

Core Libraries:

  1. opencv-python, opencv-contrib-python (for face detection and LBPH recognition)

  2. pandas (for data manipulation)

  3. tkinter (for GUI components)

  4. Standard Utilities: os, csv, datetime

📂 Project Structure
Upon running, the application automatically generates the required directory tree:

images/: Stores cropped, grayscale face datasets.

trainer/: Holds the trained model file (trainer.yml).

attendance/: Saves daily attendance reports (DD-MM-YYYY.csv).

students.csv: Master database linking Student IDs to names.

⚙️ Setup and Installation
Clone the repository:

Bash
git clone <your-repo-url>
Install the required dependencies:

Bash
pip install opencv-python opencv-contrib-python pandas
Run the main script:

Bash
python Student_Attendence.py
💡 How to Use the GUI
Enter a Student ID and Student Name, then click Register Student.

Keep the same Student ID in the entry box, look at the webcam, and click Capture Face (wait for 30 frames/images to be captured).

Click Train Data to compile all captured face images into the recognition model.

Click Start Attendance to launch the real-time scanner. Anyone registered and trained will be marked as present when their face is recognized.
