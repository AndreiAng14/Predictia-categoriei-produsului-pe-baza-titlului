# predict_category.py
import pandas as pd
import joblib
import os
from train_model import apply_feature_engineering

def main():
    model_path = 'models/product_classifier.pkl'
    
    if not os.path.exists(model_path):
        print(f"Eroare: Modelul nu există la {model_path}. Rulează train_model.py mai întâi.")
        return
        
    # Încărcăm modelul
    print("Se încarcă modelul...")
    model = joblib.load(model_path)
    
    print()
    print("Clasificator Produse Interactiv")
    print("Introdu titlul produsului pentru a afla categoria.")
    print("Scrie 'exit' pentru a ieși.\n")
    
    while True:
        user_input = input("Titlu produs: ")
        
        if user_input.lower() in ['exit', 'ieșire', 'iesire']:
            print("Închidere program. Spor!")
            break
            
        if not user_input.strip():
            print("Te rog introdu un text valid.\n")
            continue
            
        # Creăm un DataFrame cu un singur rând pentru a-l trece prin funcția noastră
        df_input = pd.DataFrame({'Product Title': [user_input]})
        
        # Extragem caracteristicile (lungime, cifre, etc.)
        df_features = apply_feature_engineering(df_input)
        
        # Facem predicția
        prediction = model.predict(df_features)[0]
        
        print(f" Categorie prezisă: {prediction}\n")

if __name__ == "__main__":
    main()