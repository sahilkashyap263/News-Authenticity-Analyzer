"""
Fake News Detection Web App
Built with Streamlit
"""

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from feature_extractor import EnhancedFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fake-news {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        model = joblib.load('improved_model.pkl')
        feature_extractor = joblib.load('improved_feature_extractor.pkl')
        return model, feature_extractor, True
    except FileNotFoundError:
        return None, None, False

# Prediction function
def predict_news(text, model, feature_extractor):
    """Make prediction with confidence scores"""
    X = feature_extractor.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    confidence = max(proba)
    fake_prob = proba[1]
    real_prob = proba[0]
    
    label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
    
    if confidence >= 0.9:
        conf_level = "Very High"
    elif confidence >= 0.75:
        conf_level = "High"
    elif confidence >= 0.6:
        conf_level = "Moderate"
    else:
        conf_level = "Low"
    
    return {
        'prediction': label,
        'confidence': confidence,
        'confidence_level': conf_level,
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }

# Create probability gauge chart
def create_gauge_chart(probability, title, color):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#e8f5e9'},
                {'range': [33, 66], 'color': '#fff9c4'},
                {'range': [66, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

# Main app
def main():
    # Load models
    model, feature_extractor, models_loaded = load_models()
    
    # Header
    st.title("🔍 Fake News Detection System")
    st.markdown("### AI-Powered News Verification Tool")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This app uses machine learning to detect fake news articles.
        
        **How it works:**
        1. Enter or paste news text
        2. Click 'Analyze'
        3. Get instant results with confidence scores
        
        **Accuracy:** ~95%
        **Model:** Ensemble (SVM + LR + RF)
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        show_metrics = st.checkbox("Show detailed metrics", value=True)
        show_charts = st.checkbox("Show probability charts", value=True)
        
        st.markdown("---")
        st.header("📈 Statistics")
        if 'total_predictions' not in st.session_state:
            st.session_state.total_predictions = 0
            st.session_state.fake_count = 0
            st.session_state.real_count = 0
        
        st.metric("Total Predictions", st.session_state.total_predictions)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fake", st.session_state.fake_count)
        with col2:
            st.metric("Real", st.session_state.real_count)
    
    # Main content
    if not models_loaded:
        st.error("⚠️ Models not found! Please train the model first.")
        st.info("Run the training script to generate model files.")
        st.code("python improved_fake_news_detection.py", language="bash")
        return
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Type/Paste Text", "📄 Upload CSV File"],
        horizontal=True
    )
    
    if input_method == "✍️ Type/Paste Text":
        # Text input
        st.subheader("📝 Enter News Article")
        
        # Example buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📰 Example: Real News"):
                st.session_state.example_text = "The Federal Reserve announced today that interest rates will remain unchanged following the monthly policy meeting. Officials cited stable inflation and steady employment growth as key factors in their decision."
        with col2:
            if st.button("🚫 Example: Fake News"):
                st.session_state.example_text = "SHOCKING! Scientists discover miracle herb that cures cancer overnight! Big Pharma HATES this simple trick that doctors don't want you to know!"
        with col3:
            if st.button("🔄 Clear"):
                st.session_state.example_text = ""
        
        # Text area
        news_text = st.text_area(
            "Paste your news article here:",
            value=st.session_state.get('example_text', ''),
            height=200,
            placeholder="Enter the news article you want to verify..."
        )
        
        # Character count
        char_count = len(news_text)
        st.caption(f"Character count: {char_count}")
        
        if char_count > 0 and char_count < 50:
            st.warning("⚠️ Text is very short. Predictions are more accurate with longer articles.")
        
        # Analyze button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if news_text.strip():
                with st.spinner("Analyzing article..."):
                    result = predict_news(news_text, model, feature_extractor)
                    
                    # Update statistics
                    st.session_state.total_predictions += 1
                    if result['prediction'] == 'FAKE NEWS':
                        st.session_state.fake_count += 1
                    else:
                        st.session_state.real_count += 1
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Prediction box
                    if result['prediction'] == 'FAKE NEWS':
                        st.error(f"### 🚫 {result['prediction']}")
                        st.markdown(f"**Confidence Level:** {result['confidence_level']} ({result['confidence']:.1%})")
                    else:
                        st.success(f"### ✅ {result['prediction']}")
                        st.markdown(f"**Confidence Level:** {result['confidence_level']} ({result['confidence']:.1%})")
                    
                    # Detailed metrics
                    if show_metrics:
                        st.markdown("### 📈 Detailed Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Overall Confidence",
                                f"{result['confidence']:.1%}",
                                delta=None
                            )
                        with col2:
                            st.metric(
                                "Fake Probability",
                                f"{result['fake_probability']:.1%}",
                                delta=None,
                                delta_color="inverse"
                            )
                        with col3:
                            st.metric(
                                "Real Probability",
                                f"{result['real_probability']:.1%}",
                                delta=None,
                                delta_color="normal"
                            )
                    
                    # Probability charts
                    if show_charts:
                        st.markdown("### 📊 Probability Visualization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_fake = create_gauge_chart(
                                result['fake_probability'],
                                "Fake News Probability",
                                "#f44336"
                            )
                            st.plotly_chart(fig_fake, use_container_width=True)
                        
                        with col2:
                            fig_real = create_gauge_chart(
                                result['real_probability'],
                                "Real News Probability",
                                "#4caf50"
                            )
                            st.plotly_chart(fig_real, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if result['confidence'] < 0.7:
                        st.warning("""
                        **⚠️ Low Confidence Prediction**
                        - This prediction has lower confidence
                        - Consider verifying from multiple trusted sources
                        - Check the author and publication credibility
                        - Look for supporting evidence and citations
                        """)
                    elif result['prediction'] == 'FAKE NEWS':
                        st.error("""
                        **🚫 Likely Fake News**
                        - Do not share without verification
                        - Check fact-checking websites (Snopes, FactCheck.org)
                        - Look for original sources
                        - Be skeptical of sensational headlines
                        """)
                    else:
                        st.success("""
                        **✅ Likely Authentic News**
                        - The article appears legitimate
                        - Still verify from original sources when possible
                        - Check publication date and relevance
                        - Consider the source's reputation
                        """)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    else:  # CSV Upload
        st.subheader("📄 Batch Analysis - Upload CSV")
        st.info("Upload a CSV file with a column containing news articles to analyze multiple articles at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Column selection
            text_column = st.selectbox(
                "Select the column containing news text:",
                options=df.columns.tolist()
            )
            
            if st.button("🔍 Analyze All Articles", type="primary"):
                with st.spinner(f"Analyzing {len(df)} articles..."):
                    results = []
                    
                    progress_bar = st.progress(0)
                    for idx, text in enumerate(df[text_column]):
                        result = predict_news(str(text), model, feature_extractor)
                        results.append({
                            'Prediction': result['prediction'],
                            'Confidence': f"{result['confidence']:.2%}",
                            'Fake_Probability': f"{result['fake_probability']:.2%}",
                            'Real_Probability': f"{result['real_probability']:.2%}"
                        })
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    st.success("✅ Analysis complete!")
                    
                    # Summary
                    st.markdown("### 📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    fake_count = (results_df['Prediction'] == 'FAKE NEWS').sum()
                    real_count = (results_df['Prediction'] == 'REAL NEWS').sum()
                    
                    with col1:
                        st.metric("Total Articles", len(df))
                    with col2:
                        st.metric("Fake News", fake_count, delta=f"{fake_count/len(df)*100:.1f}%")
                    with col3:
                        st.metric("Real News", real_count, delta=f"{real_count/len(df)*100:.1f}%")
                    
                    # Display results
                    st.markdown("### 📋 Results")
                    st.dataframe(final_df, use_container_width=True)
                    
                    # Download button
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for assistance only. Always verify news from multiple trusted sources.</p>
        <p>Built with ❤️ using Streamlit | Model Accuracy: ~95%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()