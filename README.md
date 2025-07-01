# GlycemicLoad 🩺

A Python toolkit for analyzing and estimating glycemic load (GL) of meals and dishes based on an image.

## 🧩 Project Structure

```
GlycemicLoad/
├── FoodClassification/         # Pre-trained food recognition models & utilities
├── GLEstimator/                # Glycemic load estimation code
├── PortionsEstimation/         # Portion size detection & estimation
├── DownloadFilesFromDataset_Nutrition5k.py   # Data acquisition script
├── HandlingDishMetadata.py     # Metadata parsing and cleaning
├── requirements.txt           # Python dependencies
```

## 🚀 Features

* **Food Classification**: Recognize food items and predict portion sizes from images or datasets.
* **Glycemic Load Estimation**: Compute GL using nutrition facts and estimated glycemic index.
* **Data Preparation**: Automated download and cleaning of Nutrition5k dataset components.
* **Interactive Notebooks**: Demonstrations and experiments with food models and GL calculations.

## 🔧 Installation

```sh
# Clone repository
git clone https://github.com/RotemGeva226/GlycemicLoad.git
cd GlycemicLoad

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📥 Data Setup

Download the Nutrition5k dataset using the provided script:

```sh
python DownloadFilesFromDataset_Nutrition5k.py --output-dir data/
```

This retrieves images, nutrition metadata, and food labels automatically.

## ⚙️ Usage

### 1. Food Classification

```python
from FoodClassification.classifier import FoodClassifier

clf = FoodClassifier(model_path='FoodClassification/model.pth')
predictions = clf.predict('data/image_001.jpg')
print(predictions)  # e.g., {'apple_pie': 0.85, ...}
```

### 2. Portion Estimation

```python
from PortionsEstimation.portion import PortionEstimator

est = PortionEstimator()
grams = est.estimate('data/image_001.jpg')
print(f'Estimated portion: {grams:.2f} g')
```

### 3. Glycemic Load Estimation

```python
from GLEstimator.gl import GlycemicLoadEstimator

gle = GlycemicLoadEstimator(nutrition_db='data/nutrition.json')
gl_value = gle.compute_gl(food='apple_pie', portion_g=150)
print(f'Estimated GL: {gl_value:.2f}')
```

## 📦 Requirements

Python 3.7+, plus these core dependencies:

* `numpy`
* `pandas`
* `torch` (or `tensorflow`, depending on model implementation)
* `scikit-learn`
* `opencv-python`
* `matplotlib` / `seaborn`

See `requirements.txt` for a complete list.

## 📄 License

This project is released under the [MIT License](LICENSE), feel free to use it in academic or commercial work.
