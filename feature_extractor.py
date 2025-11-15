"""
Enhanced Feature Extractor Module
This file must be in the same directory as your training script and Streamlit app
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from scipy.sparse import hstack

class EnhancedFeatureExtractor:
    """Extract both text and metadata features"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_features=5000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.7
        )
    
    def extract_metadata_features(self, texts):
        """Extract non-text features to reduce style-based overfitting"""
        features = []
        
        for text in texts:
            # Basic statistics
            word_count = len(text.split())
            char_count = len(text)
            avg_word_len = char_count / max(word_count, 1)
            
            # Punctuation analysis
            exclamation_count = text.count('!')
            question_count = text.count('?')
            quote_count = text.count('"') + text.count("'")
            
            # Capital letters (often used in sensational headlines)
            capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            
            # Sentiment analysis
            try:
                blob = TextBlob(text)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_subjectivity = blob.sentiment.subjectivity
            except:
                sentiment_polarity = 0
                sentiment_subjectivity = 0
            
            # Clickbait indicators
            clickbait_words = ['shocking', 'unbelievable', 'amazing', 'you wont believe',
                             'breaking', 'urgent', 'must see', 'instantly']
            clickbait_score = sum(word.lower() in text.lower() for word in clickbait_words)
            
            # All caps words (sensationalism indicator)
            words = text.split()
            all_caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
            
            features.append([
                word_count, avg_word_len, exclamation_count, question_count,
                capital_ratio, sentiment_polarity, sentiment_subjectivity,
                clickbait_score, all_caps_ratio, quote_count
            ])
        
        return np.array(features)
    
    def fit_transform(self, texts):
        """Combine TF-IDF and metadata features"""
        tfidf_features = self.tfidf.fit_transform(texts)
        meta_features = self.extract_metadata_features(texts)
        return hstack([tfidf_features, meta_features])
    
    def transform(self, texts):
        """Transform new texts"""
        tfidf_features = self.tfidf.transform(texts)
        meta_features = self.extract_metadata_features(texts)
        return hstack([tfidf_features, meta_features])