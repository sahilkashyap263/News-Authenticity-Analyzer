import sys
import joblib

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_label(text: str) -> str:
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "Fake (1)" if pred == 1 else "Real (0)"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your news text here"')
        sys.exit(1)
    print(predict_label(sys.argv[1]))