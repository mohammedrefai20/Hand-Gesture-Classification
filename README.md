# Hand Gesture Classification

This project presents a hand gesture classification workflow based on MediaPipe hand landmarks extracted from the HaGRID dataset. The complete implementation is provided in the Colab notebook `ML_Project.ipynb`.

## Project Steps

### Step 1: Data Loading
The workflow begins by loading the dataset and organizing the feature values and labels required for training and evaluation.

### Step 2: Data Understanding
The notebook then performs exploratory analysis to understand the dataset structure, class balance, and data quality. This includes reviewing class samples, checking class distribution, and identifying outliers.

### Step 3: Preprocessing
Landmark coordinates are normalized to reduce the impact of hand location and scale differences across samples. The preprocessing is based on landmark translation and distance-based normalization.

### Step 4: Train and Test Split
After preprocessing, the data is split into training and testing sets using an 80 to 20 ratio to measure generalization performance.

### Step 5: Model Training
The project trains and compares multiple machine learning models, including K-Nearest Neighbors, Support Vector Machine, Random Forest, and AdaBoost with a Decision Tree base model.

### Step 6: Evaluation
Model performance is evaluated using accuracy, precision, recall, and F1-score. The final model is selected according to the comparative results.

### Step 7: Model Saving
The trained models are saved to support later inference and deployment.

### Step 8: Output Video
I used this code to generate the output video, and I have uploaded it in a Google Drive link.

## Repository Files
`ML_Project.ipynb` contains the full workflow from preprocessing to final output generation.
`README.md` contains a concise overview of the project and its implementation steps.
