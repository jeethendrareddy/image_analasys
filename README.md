# Arhar-Khesari Dal Image Analysis

A machine learning project to detect and quantify the percentage of Khesari dal (lentils) mixed with Arhar dal using image analysis.

## Project Overview

This project uses deep learning-based feature extraction combined with traditional machine learning algorithms to predict the percentage of Khesari dal in Arhar dal mixtures. Multiple models and configurations have been tested to achieve optimal performance.

## Features

- **Image Augmentation**: Automated data augmentation to increase training dataset size
- **Feature Extraction**:
  - ResNet50-based feature extraction
  - DenseNet121-based feature extraction
- **Multiple ML Models**:
  - Support Vector Regression (SVR)
  - Random Forest Regressor
  - Decision Tree Regressor
- **Deep Learning**: End-to-end CNN using ResNet18 transfer learning
- **Interactive Web Apps**: Streamlit-based applications for real-time predictions

## Project Structure

```
image_analasys/
├── config.py                    # Centralized configuration for paths
├── requirements.txt             # Python dependencies
├── Codes/
│   ├── ImageAugumentation.py    # Data augmentation script
│   ├── FeatureExtraction/
│   │   ├── FeatureExtraction-7.py   # ResNet50 features
│   │   ├── FeatureExtraction-8.py   # DenseNet features (v8)
│   │   ├── FeatureExtraction-10.py  # DenseNet features (v10)
│   │   └── ReduceFeatures.py        # Feature selection
│   ├── ModelTraining/
│   │   ├── ModelTraining_8.py       # Train on ResNet features
│   │   ├── ModelTraining_9.py       # Train on DenseNet features (v9)
│   │   ├── ModelTraining_10.py      # Train on DenseNet features (v10)
│   │   ├── ModelTraining_12.py      # Train on DenseNet features (v12)
│   │   └── DeepLearningModel.py     # End-to-end CNN training
│   └── StreamLitApp/
│       ├── StreamLitApp_8.py        # Web app for Test-8 models
│       ├── StreamLitApp_9.py        # Web app for Test-9 models
│       ├── StreamLitApp_10.py       # Web app for Test-10 models
│       └── StreamLitApp_DL.py       # Web app for deep learning model
├── Data/                        # Dataset folder (images organized by percentage)
├── Features/                    # Extracted features (.npy, .csv files)
├── Models/                      # Trained ML models
│   ├── Test-8/                  # ResNet50-based models
│   ├── Test-9/                  # DenseNet-based models (C=10)
│   ├── Test-10/                 # DenseNet-based models (C=30)
│   └── Test-12/                 # DenseNet-based models (C=50)
├── DL_Models/                   # Deep learning models
│   └── Test-1/                  # CNN model
└── Results/                     # Model performance metrics
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd image_analasys
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place images in `Data/` directory
   - Organize by percentage folders: `20_Percent/`, `30_Percent/`, etc.

## Usage

### 1. Data Augmentation

Generate augmented images to increase dataset size:

```bash
cd Codes
python ImageAugumentation.py
```

### 2. Feature Extraction

Extract features using pre-trained models:

```bash
# Using DenseNet121 (recommended)
cd Codes/FeatureExtraction
python FeatureExtraction-8.py

# Using ResNet50
python FeatureExtraction-7.py
```

### 3. Train Models

Train machine learning models on extracted features:

```bash
cd Codes/ModelTraining

# Train on DenseNet features (Test-10 - best performance)
python ModelTraining_10.py

# Or train end-to-end deep learning model
python DeepLearningModel.py
```

### 4. Run Web Application

Launch the Streamlit web interface for predictions:

```bash
# Run from project root directory
cd Codes/StreamLitApp

# For Test-10 models (recommended)
streamlit run StreamLitApp_10.py

# For deep learning model
streamlit run StreamLitApp_DL.py
```

Then open your browser to `http://localhost:8501`

## Model Performance

Performance comparison across different configurations:

| Test Version | Features | Best Model | MAE | R² Score |
|-------------|----------|------------|-----|----------|
| Test-8 | ResNet50 | SVM | 5.24 | 0.90 |
| Test-9 | DenseNet | Random Forest | 2.92 | 0.92 |
| **Test-10** | **DenseNet** | **SVM** | **4.29** | **0.93** |
| Test-12 | DenseNet | SVM | 7.02 | 0.92 |

**Test-10 configuration is recommended** for the best balance of accuracy and performance.

## Model Configurations

### Test-8 (ResNet50 Features)
- Feature Extractor: ResNet50
- SVM: C=50, kernel='rbf'
- Random Forest: n_estimators=150
- Decision Tree: default parameters

### Test-9 (DenseNet Features)
- Feature Extractor: DenseNet121
- SVM: C=10, kernel='rbf'
- Random Forest: n_estimators=200
- Decision Tree: default parameters

### Test-10 (DenseNet Features) - **RECOMMENDED**
- Feature Extractor: DenseNet121
- SVM: C=30, kernel='rbf'
- Random Forest: n_estimators=250, max_depth=40
- Decision Tree: default parameters

### Test-12 (DenseNet Features)
- Feature Extractor: DenseNet121
- SVM: C=50, kernel='rbf'
- Random Forest: n_estimators=300, max_depth=40
- Decision Tree: default parameters

## Configuration

All paths and settings are centralized in `config.py`. Modify this file to:
- Change data directories
- Adjust model save locations
- Update hyperparameters

## Dataset Structure

Expected dataset organization:

```
Data/
├── 20_Percent/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 30_Percent/
│   ├── image1.jpg
│   └── ...
├── 40_Percent/
├── 50_Percent/
├── 60_Percent/
├── 70_Percent/
├── 80_Percent/
└── 90_Percent/
```

## Technologies Used

- **TensorFlow/Keras**: Pre-trained models (ResNet50, DenseNet121)
- **PyTorch**: Deep learning model training
- **scikit-learn**: Traditional ML algorithms
- **OpenCV**: Image processing
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation

## Results

Results from all test runs are saved in `Results/` directory:
- `model_comparison_Test-8.txt`: ResNet50-based models
- `DenseNet_results_9.txt`: DenseNet models (Test-9)
- `DenseNet_results_10.txt`: DenseNet models (Test-10)
- `DenseNet_results_12.txt`: DenseNet models (Test-12)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Pre-trained models from TensorFlow/Keras and PyTorch
- ImageNet dataset for transfer learning weights

## Contact

For questions or collaboration, please open an issue on GitHub.
