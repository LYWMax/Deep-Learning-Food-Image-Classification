# 🍜 Food Image Classification

A 10-class food image classifier built with **MobileNet transfer learning** using TensorFlow/Keras. The model is trained to recognise and classify food images into one of 10 predefined food categories.

---

## 📋 Food Categories
| | Food |
|---|---|
| 1 | Bibimbap |
| 2 | Ceviche |
| 3 | Chicken Curry |
| 4 | French Onion Soup |
| 5 | Gyoza |
| 6 | Oysters |
| 7 | Pancakes |
| 8 | Ramen |
| 9 | Ravioli |
| 10 | Sashimi |

---

## 📊 Model Results

| Model | Test Accuracy |
|---|---|
| Baseline (frozen MobileNet, no augmentation) | 80.8% |
| Fine-tuned MobileNet (last 3 blocks) | 88.8% |
| Fine-tuned + Extra Dense Layer (64 neurons) | 89.8% |
| **Fine-tuned + Dropout (0.4) — Best Model** | **91.8%** |

The best model is a fine-tuned MobileNet with a Dropout layer of 0.4, saved at epoch 20.

---

## 🗂️ Project Structure

```
food-image-classification/
├── Food_Image_Classification_Notebook.ipynb   # Main notebook — model building, training, evaluation
├── Image_Preprocessing.ipynb                  # Splits Food-101 dataset into train/val/test
├── 14.txt                                     # Assigned food category list
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/LYWMax/food-image-classification.git
cd food-image-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the Food-101 dataset from Kaggle:
👉 https://www.kaggle.com/kmader/food41

Extract it so you have an `images/` folder in your project directory.

### 4. Preprocess the data
Run `Image_Preprocessing.ipynb` to split the images into:
- `train/` — 750 images per food category
- `validation/` — 200 images per food category
- `test/` — 50 images per food category

### 5. Run the main notebook
Open and run `Food_Image_Classification_Notebook.ipynb` in order.

---

## 🧠 Model Architecture

The model uses **MobileNet** as a convolutional base (pre-trained on ImageNet) with a custom classifier head:

```
MobileNet (conv base, partially fine-tuned)
    → Flatten
    → Dense(128, activation='relu')
    → Dropout(0.4)
    → Dense(10, activation='softmax')
```

**Training details:**
- Image size: 224 × 224
- Batch size: 20
- Optimizer: RMSprop (lr=1e-4)
- Loss: Categorical Crossentropy
- Epochs: 30 with ModelCheckpoint

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Keras](https://img.shields.io/badge/Keras-2.10-red)

- Python 3.10
- TensorFlow / Keras 2.10
- NumPy, Pandas, Matplotlib
- Scikit-learn
- Pillow
