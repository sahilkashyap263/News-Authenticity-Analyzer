# Fake News Detection System 📰🚫

Detect whether a news article is **real** or **fake** using Python, Scikit-learn, and TF-IDF. Super simple, super fast, and hella accurate.  

---

## 🔧 Tech Stack
- **Python 3**
- **Pandas & NumPy** – data handling  
- **Scikit-learn** – machine learning magic  
- **TF-IDF Vectorizer** – turning text into numbers  
- **LinearSVC** – classifier for real vs fake  
- **Joblib** – save & load your model  

---

## 🗂 Project Structure
Fake-News-Detection/
│
├─ True.csv # Real news dataset
├─ Fake.csv # Fake news dataset
├─ model.pkl # Saved trained model
├─ vectorizer.pkl # Saved TF-IDF vectorizer
├─ main.py # Training & evaluation script
└─ predict.py # Predict single news text from command line


---

## 🚀 How It Works
1. Load datasets (`True.csv` & `Fake.csv`)  
2. Label encoding: `0 = Real`, `1 = Fake`  
3. Clean and merge data, combine `title + text`  
4. Convert text to TF-IDF features  
5. Split data into train/test sets  
6. Train `LinearSVC` classifier  
7. Evaluate model accuracy & classification report  
8. Save model and vectorizer for later use  

---

## 💻 Usage

### Train the model
```bash
python main.py

Outputs:

model.pkl

vectorizer.pkl

Accuracy & classification report

Predict a news article
python predict.py "Your news text here"

python predict.py "NASA confirms water found on Mars in large quantities."
# Output: Real (0)

python predict.py "Scientists claim a secret herb cures every disease overnight."
# Output: Fake (1)

📊 Model Performance

Accuracy: ~99.6%

Class balance: Real: 21417 | Fake: 23481

Super fast predictions – perfect for integrating into apps or pipelines

⚡ Notes

Make sure True.csv and Fake.csv are in the correct path

Model & vectorizer must be loaded for predictions

Text preprocessing is minimal: just lowercasing, removing NaNs, combining title + text

💡 Future Improvements

Add deep learning models like LSTM or BERT for higher accuracy

Include more datasets for better generalization

Build a simple web app for real-time news verification

Made with ❤️ by Sahil Kashyap
