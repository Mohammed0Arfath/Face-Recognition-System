# Advanced Face Recognition System: A Comparative Analysis

This repository contains a comprehensive, end-to-end face recognition system developed in a Jupyter Notebook. The project goes beyond a simple implementation by systematically comparing four state-of-the-art face embedding techniques and five different machine learning classifiers to identify the most robust and accurate combination. The system achieves up to **98% accuracy** and includes a functional prototype for real-time face recognition from a live webcam feed.

## Key Features

-   **Comparative Analysis:** Rigorous evaluation of four distinct face detection and embedding pipelines:
    1.  **OpenCV YuNet + `imgbeddings`**
    2.  **`face_recognition` Library (dlib/ResNet)**
    3.  **InsightFace (ArcFace)**
    4.  **FaceNet (MTCNN + InceptionResnetV1)**
-   **Multiple Classifiers:** Training and evaluation of five classical ML models: **KNN**, **SVM**, **MLP**, **AdaBoost**, and **LDA**.
-   **High Accuracy:** Achieves a top accuracy of **98%** on the test set, demonstrating the effectiveness of modern embedding techniques.
-   **Anomaly Detection:** Implements a **One-Class SVM** and distance thresholding to intelligently identify and label "Unknown" faces that are not part of the training dataset.
-   **Live Webcam Demo:** An interactive, in-notebook prototype using JavaScript to stream video from a webcam and perform real-time recognition.
-   **In-depth Visualization:** Detailed performance analysis using PCA, t-SNE, Confusion Matrices, ROC Curves, and Precision-Recall Curves.

## Live Demo

The final part of the notebook features a real-time application that uses your webcam to detect and identify faces, drawing bounding boxes and displaying the predicted identity with a confidence score.

![image](https://github.com/user-attachments/assets/a0715740-7372-46d5-8cfe-5e0efdbf5e0c)

*Caption: Real-time face recognition using the trained InsightFace + LDA model.*

## System Architecture

The project is structured as a multi-phase pipeline, allowing for modular experimentation and clear evaluation.

1.  **Feature Extraction:** An image dataset is processed by one of the four embedding methods. Each method detects the face, crops it, and generates a high-dimensional feature vector (embedding).
2.  **Model Training:** The generated embeddings and their corresponding labels are used to train five different classifiers.
3.  **Performance Evaluation:** Each trained model is evaluated on an unseen test set to measure its performance using a variety of metrics.
4.  **Real-time Inference:** The best-performing models are loaded and used in a real-time application to identify faces from a live video stream or static images.

   ![Face_Recognition_System_Architecture_Diagram](https://github.com/user-attachments/assets/a3366591-cb26-4895-b50c-efc10130b3d8)


```
Image Dataset -> [Face Detection & Embedding] -> Embedding Vector -> [ML Classifier] -> Predicted Identity
```

## Models & Performance

The following table summarizes the test accuracy for each combination of embedding method and classifier.

| Embedding Method | KNN | SVM | MLP | AdaBoost | **LDA** | **Best Performer** |
| :--- | :-: | :-: | :-: | :---: | :---: | :---: |
| **1. `imgbeddings`** | 94% | 96% | **97%** | 12% | **97%** | MLP / LDA |
| **2. `face_recognition`**| 98% | 98% | 98% | 11% | **98%** | **LDA / SVM / MLP** |
| **3. InsightFace (ArcFace)**| 98% | 98% | 98% | 16% | **98%** | **LDA / SVM / MLP**|
| **4. FaceNet** | 91% | 91% | 91% | 33% | **92%** | LDA |

### Key Findings

-   **Best Embeddings:** **InsightFace (ArcFace)** and **`face_recognition`** produced the highest quality embeddings, enabling the classifiers to achieve exceptional accuracy.
-   **Best Classifiers:** **Linear Discriminant Analysis (LDA)**, **MLP**, and **SVM** consistently delivered top-tier performance, demonstrating their suitability for classifying high-dimensional feature vectors.
-   **Optimal Combination:** The combination of **InsightFace embeddings and an LDA classifier** is highly recommended for its state-of-the-art accuracy and the computational efficiency of LDA.

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` is recommended. Based on the notebook, the key libraries are:
    ```bash
    pip install opencv-python scikit-learn matplotlib seaborn pandas numpy
    pip install insightface onnxruntime
    pip install face_recognition
    pip install facenet-pytorch torch torchvision
    pip install jupyter
    ```

4.  **Download the Dataset:**
    The project uses the [Face Recognition Dataset](https://www.kaggle.com/datasets/ashwingupta3012/face-recognition-dataset) from Kaggle. Download it and place the `Original Images` folder in a location accessible by the notebook.

5.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## Usage

1.  **Open the Notebook:** Open the `face-recognition.ipynb` file in Jupyter.

2.  **Update Paths:** Make sure to update the file paths for the dataset and the directories where models and embeddings will be saved (e.g., Google Drive paths should be changed to local paths).

3.  **Run the Cells Sequentially:**
    -   **Phase 1 (Feature Extraction):** The notebook contains four separate sections for generating embeddings. You only need to run **one** of these sections to create a set of embeddings. The **InsightFace (ArcFace)** section is highly recommended.
    -   **Phase 2 (Model Training):** Run the corresponding training and evaluation cells for the embedding set you generated. This will train all five classifiers and save them as `.pkl` files.
    -   **Phase 3 (Visualization):** These cells will generate plots like t-SNE and confusion matrices to help you analyze the results.
    -   **Phase 4 (Real-time Inference):** Run the final sections to test the system on individual images or launch the interactive webcam demo.

## Project Structure

```
.
├── face-recognition.ipynb      # The main Jupyter Notebook
├── data/                       # Stores embeddings from Method 1
│   ├── face_data.pkl
│   └── labels.pkl
├── data2/                      # Stores embeddings from Method 2
├── data3/                      # Stores embeddings from Method 3 (Recommended)
├── data4/                      # Stores embeddings from Method 4
├── models/                     # Stores trained classifiers for data/
│   ├── knn_model.pkl
│   └── ...
├── models2/                    # etc.
├── models3/
├── models4/
├── test_data/                  # Place test images here for inference
└── README.md
```

## Deployment

The notebook contains a sophisticated prototype for web deployment using JavaScript. For a production-ready application, this can be extended by:
1.  **Creating a Backend API:** Use a web framework like **Flask** or **FastAPI** to wrap the model inference code. The API would accept an image and return the predicted identities and bounding box coordinates as a JSON response.
2.  **Building a Frontend:** Develop a web interface using a framework like React or Vue.js that communicates with the backend API and renders the results on the user's video feed.
3.  **Containerization:** Use Docker to containerize the application for easy and consistent deployment on cloud services like AWS, GCP, or Azure.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

-   This project utilizes the [Face Recognition Dataset](https://www.kaggle.com/datasets/ashwingupta3012/face-recognition-dataset) available on Kaggle.
-   Credit to the developers of the `face_recognition`, `InsightFace`, and `facenet-pytorch` libraries for their invaluable open-source contributions.
