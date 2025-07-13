# GlycemicLoad 🩺

A Python toolkit for analyzing and estimating glycemic load (GL) of meals and dishes based on an image.

## 🗂️ Project Structure

```
GlycemicLoad/
├── FoodClassification/         # Pre-trained food recognition models & utilities
├── GLEstimator/                # Glycemic load estimation code
├── PortionsEstimation/         # Portion size detection & estimation
├── DownloadFilesFromDataset_Nutrition5k.py   # Data acquisition script
├── HandlingDishMetadata.py     # Metadata parsing and cleaning
├── requirements.txt           # Python dependencies
```

## 🔧 Installation

```sh
# Clone repository
git clone https://github.com/RotemGeva226/GlycemicLoad.git
cd GlycemicLoad

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

Python 3.7+, plus these core dependencies:

* `numpy`
* `pandas`
* `torch`
* `scikit-learn`
* `opencv-python`
* `matplotlib` / `seaborn`

See `requirements.txt` for a complete list.
