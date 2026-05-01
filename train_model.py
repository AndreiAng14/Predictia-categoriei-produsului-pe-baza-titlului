# train_model.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import os

# Definim funcția aici pentru a putea fi importată de scriptul de testare
def apply_feature_engineering(df):
    data = df.copy()
    data['Product Title'] = data['Product Title'].astype(str)
    data['char_count'] = data['Product Title'].apply(len)
    data['word_count'] = data['Product Title'].apply(lambda x: len(x.split()))
    data['digit_count'] = data['Product Title'].apply(lambda x: sum(c.isdigit() for c in x))
    return data

def main():
    print("Inițializare Antrenare Model")
    
    # 1. Încărcare date
    filepath = 'data/products.csv'
    if not os.path.exists(filepath):
        print(f"Eroare: Nu găsesc fișierul {filepath}")
        return
        
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Product Title', 'Category Label']).copy()
    
    print(f"Date încărcate. Rânduri valide: {len(df)}")
    
    # 2. Aplicare Feature Engineering
    df = apply_feature_engineering(df)
    
    X = df[['Product Title', 'char_count', 'word_count', 'digit_count']]
    y = df['Category Label']
    
    # 3. Configurare Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1,2), max_features=10000), 'Product Title'),
            ('num', StandardScaler(), ['char_count', 'word_count', 'digit_count'])
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(random_state=42, dual='auto'))
    ])
    
    # 4. Antrenare
    print("Se antrenează modelul... (ar putea dura câteva momente)")
    pipeline.fit(X, y)
    
    # 5. Salvare model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/product_classifier.pkl')
    print(" Model antrenat și salvat cu succes în 'models/product_classifier.pkl'!")

if __name__ == "__main__":
    main()