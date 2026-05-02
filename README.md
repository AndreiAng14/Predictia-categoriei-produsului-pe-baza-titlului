# Product Classification Pipeline 

# Cum se utilizează:

## 1. Antrenarea modelului
Asigură-te că fișierul products.csv se află în folderul data/. Apoi rulează:
    ```python python train_model.py```

## 2. Testarea Interactivă
După ce modelul este antrenat, îl poți testa live introducând titluri de produse de la tastatură:
   ```python python predict_category.py ```
    
Acest proiect conține un model de Machine Learning end-to-end pentru clasificarea automată a produselor în funcție de titlul lor. 

## 📂 Structura Proiectului
```text
├── data/                      # Aici se află products.csv
├── models/                    # Aici se va salva modelul .pkl
├── notebooks/                 # Experimente, EDA și evaluare (Jupyter)
├── train_model.py             # Scriptul de antrenare a modelului final
├── predict_category.py        # Script interactiv de testare a predicțiilor
└── README.md                  # Documentația proiectului

