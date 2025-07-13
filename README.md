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

## 📦 Requirements

Python 3.7+, plus these core dependencies:

* `numpy`
* `pandas`
* `torch` (or `tensorflow`, depending on model implementation)
* `scikit-learn`
* `opencv-python`
* `matplotlib` / `seaborn`

See `requirements.txt` for a complete list.
