# 🎵 SongSavvy - Hindi Bollywood Song Recommender

**SongSavvy** is a content-based recommendation system for Hindi Bollywood songs. It analyzes features such as artist, genre, mood, lyrics, and other metadata to recommend songs similar to your selection.

---

## ✨ Features

- 🔍 **Search Songs**: Search for songs by title.
- 🎨 **Filter Options**: Filter songs by artist, mood, and year range.
- 🎧 **Personalized Recommendations**: Get recommendations based on selected song similarity.
- 🔍 **Explainability**: See why a song was recommended through feature-based similarity.

---


## 📁 Project Structure

```
SongSavvy/
├── app.py                     # Streamlit app
├── recommendation_engine.py  # Core recommendation logic
├── utils.py                  # Preprocessing and helper functions
├── data/
│   └── bollywood_songs.csv   # Dataset
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata (optional)
├── .gitignore                # Files to ignore in Git
└── README.md                 # Project documentation
```

---

## 🛠️ Installation & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/SongSavvy.git
   cd SongSavvy
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Enjoy discovering music!** 🎶

---

## 📊 Dataset

The app uses a curated dataset of Bollywood songs:  
**Path:** `data/bollywood_songs.csv`  
Make sure this file exists before running the application.

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   Song metadata is cleaned and prepared using `utils.py`.

2. **Feature Extraction**  
   - TF-IDF vectorization is applied on song features (lyrics, artist, genre, etc.)

3. **Similarity Measurement**  
   - Cosine similarity is calculated between song vectors.

4. **Recommendations**  
   - Top-N most similar songs are shown with explanation of similarity.

---

## 🧰 Dependencies

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- scikit-learn

---



