# Fake News Detection System 📰

A machine learning-powered system to classify news articles as real or fake using Natural Language Processing and supervised learning techniques.

## Overview

This project uses TF-IDF vectorization and a Linear Support Vector Classifier to detect fake news with approximately 99.6% accuracy. The system processes news article text and titles to make predictions in real-time.

## Features

- **High Accuracy**: Achieves ~99.6% classification accuracy
- **Fast Predictions**: Optimized for quick inference
- **Simple API**: Easy-to-use command-line interface
- **Persistent Models**: Save and load trained models for deployment
- **Minimal Dependencies**: Built with standard ML libraries

## Tech Stack

- **Python 3**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning framework
- **TF-IDF Vectorizer**: Text feature extraction
- **LinearSVC**: Support Vector Machine classifier
- **Joblib**: Model serialization

## Project Structure

```
Fake-News-Detection/
│
├── True.csv           # Real news dataset (21,417 articles)
├── Fake.csv           # Fake news dataset (23,481 articles)
├── model.pkl          # Trained classifier model
├── vectorizer.pkl     # Fitted TF-IDF vectorizer
├── main.py            # Model training and evaluation
└── predict.py         # Prediction script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fake-News-Detection
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

3. Ensure datasets are in the project directory:
   - `True.csv`
   - `Fake.csv`

## Usage

### Training the Model

Run the training script to build and evaluate the model:

```bash
python main.py
```

This will:
- Load and preprocess the datasets
- Train the LinearSVC classifier
- Display accuracy metrics and classification report
- Save `model.pkl` and `vectorizer.pkl`

### Making Predictions

Use the prediction script to classify new articles:

```bash
python predict.py "Your news article text here"
```

**Examples:**

```bash
# Real news example
python predict.py "NASA announces discovery of water ice on Mars through satellite imagery."
# Output: Real (0)

# Fake news example
python predict.py "Scientists discover miracle herb that cures all diseases instantly."
# Output: Fake (1)
```

## How It Works

1. **Data Loading**: Import real and fake news datasets
2. **Labeling**: Assign labels (0 = Real, 1 = Fake)
3. **Preprocessing**: Combine article titles and text, handle missing values
4. **Vectorization**: Convert text to TF-IDF numerical features
5. **Training**: Split data (train/test) and fit LinearSVC model
6. **Evaluation**: Calculate accuracy and generate classification metrics
7. **Persistence**: Save model and vectorizer for future predictions

## Model Performance

- **Accuracy**: ~99.6%
- **Training Set**: 44,898 articles (21,417 real, 23,481 fake)
- **Features**: TF-IDF word vectors with default parameters
- **Classifier**: Linear Support Vector Classifier

## Important Notes

- Ensure `True.csv` and `Fake.csv` are in the project root directory
- Both `model.pkl` and `vectorizer.pkl` must be present for predictions
- Text preprocessing is minimal (lowercasing, NaN removal, title-text concatenation)
- The model performs best on news articles similar to the training data

## Future Enhancements

- [ ] Implement deep learning models (LSTM, BERT) for improved accuracy
- [ ] Expand dataset diversity for better generalization
- [ ] Build a web interface for real-time verification
- [ ] Add data augmentation techniques
- [ ] Implement cross-validation for robust evaluation
- [ ] Create REST API for integration with other applications

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Author

**Sahil Kashyap**

---

Made with ❤️ for combating misinformation
