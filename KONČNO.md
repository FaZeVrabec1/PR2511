# Analiza aplikacij Google Play Store

## Podatki

Podatke za analizo, uporabljene v tej seminarski nalogi, smo večinoma zbrali sami iz aplikacij v trgovini Google Play. Za primerjavo trendov skozi čas smo uporabili tudi [javno dostopen podatkovni set](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps) s portala Kaggle, ki je bil zbran leta 2021 in smo ga pridobili 15/04/2025.


**Opis atributov v podatkovni zbirki:**

| Atribut                        | Opis                                                                                   |
|---------------------------------|----------------------------------------------------------------------------------------|
| **Ime aplikacije (App Name)**   | Ime aplikacije.                                                                        |
| **ID aplikacije (App Id)**      | Edinstven identifikator aplikacije (ime paketa).                                       |
| **Kategorija (Category)**       | Kategorija aplikacije (npr. Orodja, Izobraževanje, Finance).                          |
| **Ocena (Rating)**              | Povprečna uporabniška ocena (število med 0 in 5; lahko manjka).                        |
| **Število ocen (Rating Count)** | Število uporabniških ocen (lahko manjka).                                              |
| **Število namestitev (Installs)** | Skupno število namestitev aplikacije (celo število).                                 |
| **Brezplačna (Free)**           | Ali je aplikacija brezplačna (True/False).                                             |
| **Cena (Price)**                | Cena aplikacije (decimalno število; 0.0, če je brezplačna).                            |
| **Valuta (Currency)**           | Valuta cene (običajno USD).                                                            |
| **ID razvijalca (Developer Id)**| Edinstven identifikator razvijalca.                                                    |
| **Datum izdaje (Released)**     | Datum prve izdaje aplikacije (besedilo; lahko manjka).                                 |
| **Zadnja posodobitev (Last Updated)** | Datum zadnje posodobitve (besedilo).                                             |
| **Starostna omejitev (Content Rating)** | Primernost vsebine glede na starost (npr. Za vse, Mature).                   |
| **Vsebuje oglase (Ad Supported)** | Ali aplikacija vsebuje oglase (True/False).                                         |
| **Nakupe v aplikaciji (In App Purchases)** | Ali aplikacija omogoča nakupe v aplikaciji (True/False).                   |
| **Čas zajema podatkov (Scraped Time)** | Datum in čas, ko so bili podatki zbrani.                                        |

## Izvedene analize

Najprej smo podatke analizirali po kategorijah aplikacij, kjer smo izpostavili:
- najbolj in najmanj uporabljene,
- najbolj in najmanj prenesene,
- ter najbolje in najslabše ocenjene aplikacije.

Nato smo se osredotočili na načine monetizacije (npr. brezplačne aplikacije, aplikacije z oglasi, z nakupi v aplikaciji) ter analizirali njihovo razporeditev in povprečne ocene.

![alt text](Images/Distribution_od_Monetization_Types)

V nadaljevanju smo izvedli analizo naklonjenosti (bias), kjer smo določili izjemne aplikacije in jih primerjali z ostalimi – tako po kategorijah kot po načinu monetizacije.

![alt text](Images/Bias_in_Catagories.png)
![alt text](Images/Bias_in_Monetization.png)

Izdelali smo tudi napovedovalna modela (RandomForestRegressor), ki napovedujeta, kakšno oceno bo imela aplikacija oziroma koliko prenosov lahko pričakujemo glede na izbrane lastnosti. Interaktiven način uporabe modela za napovedovanje je na voljo v Streamlit aplikaciji.

Na koncu smo rezultate primerjali še s starimi podatki iz Google Play Store iz leta 2021, da bi ugotovili trende in spremembe skozi čas.
![alt text](Images/Categories_install_change.png)
![alt text](Images/Catagories_rating_change.png)

---
**Koda:**

Koda uporabljena za analizo podatkov `project - streamlit.ipynb`.

**Streamlit aplikacija:**  
Interaktivni vpogled v rezultate in/ali model na GitHubu v mapi `/Streamlit`.
