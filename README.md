# 🔍 Fake News Detection System

An AI-powered web application that detects fake news using machine learning with 95% accuracy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📸 Screenshots

### Single Article Analysis
Analyze individual news articles with confidence scores and visual probability charts.

### Batch Processing
Upload CSV files to analyze multiple articles at once.

---

## ✨ Features

- 🎯 **Single Article Analysis** - Paste any news article for instant verification
- 📊 **Confidence Scores** - Get detailed probability breakdowns
- 📈 **Visual Charts** - Interactive gauge charts for easy interpretation
- 📄 **Batch Processing** - Upload CSV files to analyze multiple articles
- 💡 **Smart Recommendations** - Actionable advice based on results
- 📱 **Mobile Friendly** - Responsive design works on all devices
- ⚡ **Real-time Analysis** - Get results in seconds

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download training data**
   
   Download the datasets and place them in the project root:
   - [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) - Fake news articles
   - [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) - Real news articles

4. **Train the model**
   ```bash
   python improved_fake_news_detection.py
   ```
   
   This will generate:
   - `improved_model.pkl` (trained model)
   - `improved_feature_extractor.pkl` (feature extractor)
   - `confusion_matrix.png` (evaluation plot)
   - `roc_curve.png` (ROC curve plot)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

---

## 📖 Usage

### Analyze Single Article

1. Select **"Type/Paste Text"** mode
2. Click example buttons or paste your own article
3. Click **"Analyze Article"**
4. View results with confidence scores and recommendations

### Batch Analysis

1. Select **"Upload CSV File"** mode
2. Upload a CSV file containing news articles
3. Select the column with article text
4. Click **"Analyze All Articles"**
5. Download results as CSV

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.8+
- **ML Framework:** Scikit-learn
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy

### Machine Learning Model

- **Algorithm:** Ensemble (Voting Classifier)
  - Linear SVM with calibration
  - Logistic Regression
  - Random Forest
- **Features:** 
  - TF-IDF vectors (5,000 features, trigrams)
  - Metadata features (sentiment, punctuation, clickbait indicators)
- **Accuracy:** ~95% on test set
- **Training Data:** 45,000 news articles

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.0% |
| Precision (Fake) | 96.2% |
| Recall (Fake) | 96.2% |
| F1-Score (Fake) | 96.2% |
| Precision (Real) | 95.8% |
| Recall (Real) | 95.8% |
| F1-Score (Real) | 95.8% |

---

## 📁 Project Structure

```
fake-news-detection/
├── app.py                              # Streamlit web application
├── feature_extractor.py                # Feature extraction class
├── improved_fake_news_detection.py     # Model training script
├── improved_predict.py                 # Command-line prediction script
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── .gitignore                         # Git ignore rules
│
├── improved_model.pkl                  # Trained model (generated)
├── improved_feature_extractor.pkl      # Feature extractor (generated)
├── confusion_matrix.png                # Model evaluation (generated)
├── roc_curve.png                       # ROC curve (generated)
│
├── Fake.csv                           # Fake news dataset (download)
└── True.csv                           # Real news dataset (download)
```

---

## 🔧 Configuration

### Requirements

```txt
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
plotly==5.17.0
textblob==0.17.1
scipy==1.11.3
```

### Customization

Edit `.streamlit/config.toml` to customize the app theme:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

---

## 🎯 How It Works

1. **Text Preprocessing**
   - Combines article title and body text
   - Removes stop words and normalizes text
   
2. **Feature Extraction**
   - **TF-IDF Features:** Captures important words and phrases
   - **Metadata Features:** 
     - Word count, average word length
     - Punctuation patterns (!, ?, quotes)
     - Capital letter ratios
     - Sentiment analysis (polarity, subjectivity)
     - Clickbait indicators
     - All-caps word frequency

3. **Ensemble Prediction**
   - Three models vote on the final prediction
   - Probabilities are averaged for confidence scores
   - Calibrated outputs provide reliable probabilities

4. **Result Interpretation**
   - Confidence levels: Very High (>90%), High (>75%), Moderate (>60%), Low (<60%)
   - Visual gauge charts show probability distribution
   - Smart recommendations based on confidence and prediction

---

## 🎓 Dataset

This project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

**Dataset Statistics:**
- **Total Articles:** 44,898
- **Real News:** 21,417 articles
- **Fake News:** 23,481 articles
- **Topics:** Politics, world news
- **Time Period:** 2015-2018

**Citation:**
```
Clément Bisaillon. (2020). Fake and Real News Dataset. Kaggle.
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🐛 Troubleshooting

### Issue: "Models not found"
**Solution:** Run the training script first:
```bash
python improved_fake_news_detection.py
```

### Issue: "ModuleNotFoundError"
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Memory error during training"
**Solution:** Reduce the dataset size or use fewer features:
```python
max_features=3000  # in feature_extractor.py
```

### Issue: "Port already in use"
**Solution:** Use a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset: [Clément Bisaillon's Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Inspiration: Combating misinformation in the digital age
- Built with [Streamlit](https://streamlit.io/)

---

## ⚠️ Disclaimer

This tool is designed for **educational and research purposes only**. It should be used as an assistance tool, not as the sole method for verifying news authenticity. Always:

- Cross-reference with multiple trusted sources
- Check the original source and publication date
- Verify claims with fact-checking websites
- Use critical thinking when consuming news

**The model is not perfect and may make mistakes.** Do not rely solely on this tool for important decisions.

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an [Issue](https://github.com/yourusername/fake-news-detection/issues)
3. Contact the author

---

## 🗺️ Roadmap

- [ ] Add support for multiple languages
- [ ] Implement real-time news scraping
- [ ] Add source credibility checking
- [ ] Integrate fact-checking APIs
- [ ] Deploy as a browser extension
- [ ] Add user authentication and history
- [ ] Implement advanced NLP models (BERT, GPT)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ and Python**