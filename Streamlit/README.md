# Dokumentacija aplikacije Streamlit

## Pregled projekta  
Ta projekt je Streamlit aplikacija, ki analizira in vizualizira podatke iz Google Play trgovine. Aplikacija ponuja vpoglede v kategorije aplikacij, ocene in vrste monetizacije na podlagi podatkovne zbirke `New-data.csv` in `Google-Playstore_filtered.csv`. Za pričakovanje prenosov in ocene pa se uporabljata modela `final_random_forest_model_rating.pkl` in `final_random_forest_model_installs.pkl` če modela ne obstajata jih bo app.py naredil in shranil.

## Struktura projekta  
```
Streamlit
├── app.py                              # Glavna koda aplikacije Streamlit
├── Data
│   ├── New-data.csv                    # CSV podatkovni vir za analizo
│   └── Google-Playstore_filtered.csv   # CSV podatkovni vir za analizo
├── final_random_forest_model_rating.pkl    # Model za napoved ocen (Rating)
├── final_random_forest_model_installs.pkl  # Model za napoved namestitev (Installs)
├── requirements.txt                    # Seznam zahtevanih Python paketov
└── README.md                           # Dokumentacija projekta
```

## Navodila za namestitev  
1. Klonirajte repozitorij ali prenesite projektne datoteke.  
2. V terminalu se pomaknite v imenik projekta.

## Nastavitev  
Za nastavitev projekta morate namestiti zahtevane Python pakete. To lahko storite z naslednjim ukazom:

```
pip install -r requirements.txt
```

## Zagon aplikacije  
Za zagon Streamlit aplikacije izvedite naslednji ukaz v terminalu:

```
streamlit run app.py
```

S tem boste zagnali Streamlit strežnik, aplikacijo pa si boste lahko ogledali v spletnem brskalniku na naslovu `http://localhost:8501`.

## Zahteve  
- Python 3.8 ali novejši
- Glej `requirements.txt` za seznam potrebnih knjižnic.




