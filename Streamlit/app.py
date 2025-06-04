import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Google Play Store Analysis", layout="wide")
st.title("Analiza aplikacij Google Play Store")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "Category Counts, Ratings & Installs",
    "Monetization Types",
    "Biases in Categories & Monetization",
    "Prediction Model",
    "Primerjava starih in novih podatkov",
    "Raw Data"
])

@st.cache_data
def load_data_new():
    return pd.read_csv("Data/New-data.csv")
@st.cache_data
def load_data_old():
    return pd.read_csv("Data/Google-Playstore_filtered.csv")

store = load_data_new()
old_store = load_data_old()

def compute_pricing_bias(pricing_groups, installs_col='Installs', rating_col='Rating', app_id_col='App Id', bias_multiplier=2.5, min_apps_required=5):
    top_avg_all_avg_pricing = {}
    for pricing_method, df_group in pricing_groups.items():
        if len(df_group) < min_apps_required:
            continue
        df_sorted = df_group.sort_values(installs_col, ascending=False)
        avg_all_installs = df_sorted[installs_col].mean()
        standout_apps = df_sorted[df_sorted[installs_col] > bias_multiplier * avg_all_installs]
        if standout_apps.empty:
            continue
        remaining_apps = df_sorted[~df_sorted[app_id_col].isin(standout_apps[app_id_col])]
        if remaining_apps.empty:
            continue
        standout_ratings = standout_apps[rating_col].dropna()
        remaining_ratings = remaining_apps[rating_col].dropna()
        avg_top_rating = round(standout_ratings.mean(), 2) if not standout_ratings.empty else "n/a"
        avg_remaining_rating = round(remaining_ratings.mean(), 2) if not remaining_ratings.empty else "n/a"
        standout_installs = standout_apps[installs_col]
        remaining_installs = remaining_apps[installs_col]
        avg_top_installs = round(standout_installs.mean())
        avg_remaining_installs = round(remaining_installs.mean())
        total_top_installs = standout_installs.sum()
        total_remaining_installs = remaining_installs.sum()
        total_apps = len(standout_apps) + len(remaining_apps)
        Remaining_ratio = round(len(remaining_apps) / total_apps * 100, 1)
        bias_ratio = avg_top_installs / avg_remaining_installs if avg_remaining_installs > 0 else 0
        top_avg_all_avg_pricing[pricing_method] = [
            bias_ratio, Remaining_ratio, len(standout_apps), len(remaining_apps),
            avg_top_rating, avg_remaining_rating, total_top_installs, total_remaining_installs
        ]
    return top_avg_all_avg_pricing

store['Installs'] = pd.to_numeric(store['Installs'], errors='coerce')
category_counts = store['Category'].value_counts()
category_avg_installs = (
    store[store['Installs'].notna()]
    .groupby('Category')['Installs']
    .mean()
    .round(0)
    .astype(int)
    .to_dict()
)
category_avg_installs_list = list(category_avg_installs.items())
category_avg_rating = (
    store[store['Rating'].notna()]
    .groupby('Category')['Rating']
    .mean()
    .round(2)
    .to_dict()
)
category_avg_rating_list = list(category_avg_rating.items())

filtered = store[(store['Rating'].notna()) & (store['Rating'] != 0.0) & (store['Rating'] < 5.0)]
is_paid = ~filtered['Free']
has_ads = filtered['Ad Supported'] == True
has_iap = filtered['In App Purchases'] == True
fully_free = filtered['Free'] & ~has_ads & ~has_iap
with_ads = filtered['Free'] & has_ads & ~has_iap
with_iap = filtered['Free'] & ~has_ads & has_iap
with_both = filtered['Free'] & has_ads & has_iap

rating_price = {
    'Fully Free': filtered[fully_free]['Rating'].tolist(),
    'With Ads': filtered[with_ads]['Rating'].tolist(),
    'Paid': filtered[is_paid]['Rating'].tolist(),
    'With IAP': filtered[with_iap]['Rating'].tolist(),
    'With Ads + IAP': filtered[with_both]['Rating'].tolist()   
}

# --- Overview ---
if section == "Overview":
    st.markdown("""
    ### Pregled projekta

    Ta Streamlit aplikacija omogoča interaktivno analizo podatkov iz trgovine Google Play. Glavne funkcionalnosti vključujejo:
                
    - **Category Counts, Ratings & Installs:** Prikazuje najbolj in najmanj zastopane kategorije, povprečne ocene in povprečno število namestitev po kategorijah.
    - **Monetization Types:** Analizira porazdelitev in vpliv različnih strategij monetizacije, kot so brezplačne in plačljive aplikacije, oglasi ter nakupe v aplikaciji.
    - **Prediction Model:** Omogoča napoved ocene ali števila namestitev aplikacije na podlagi izbranih lastnosti.
    - **Primerjava starih in novih podatkov:** Primerja trende med podatki iz leta 2021 in novejšimi podatki.
    - **Biases in Categories & Monetization:** Odkrije izstopajoče aplikacije in posebnosti znotraj posameznih kategorij ali monetizacijskih skupin.
    - **Raw Data:** Omogoča vpogled v surove podatke, uporabljene v analizi.

    """)
    st.markdown("""
    ### Opis podatkov

    Podatke za analizo, uporabljene v tej seminarski nalogi, smo večinoma zbrali sami iz aplikacij v trgovini Google Play.            
    Za primerjavo trendov skozi čas smo uporabili tudi [javno dostopen podatkovni set](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps) s portala Kaggle, ki je bil zbran leta 2021 in smo ga pridobili 15/04/2025.

    - **Ime aplikacije (App Name):** Ime aplikacije.
    - **ID aplikacije (App Id):** Edinstven identifikator aplikacije (ime paketa).
    - **Kategorija (Category):** Kategorija aplikacije (npr. Orodja, Izobraževanje, Finance).
    - **Ocena (Rating):** Povprečna uporabniška ocena (število med 0 in 5; lahko manjka).
    - **Število ocen (Rating Count):** Število uporabniških ocen (lahko manjka).
    - **Število namestitev (Installs):** Skupno število namestitev aplikacije (celo število).
    - **Brezplačna (Free):** Ali je aplikacija brezplačna (True/False).
    - **Cena (Price):** Cena aplikacije (decimalno število; 0.0, če je brezplačna).
    - **Valuta (Currency):** Valuta cene (običajno USD).
    - **ID razvijalca (Developer Id):** Edinstven identifikator razvijalca.
    - **Datum izdaje (Released):** Datum prve izdaje aplikacije (besedilo; lahko manjka).
    - **Zadnja posodobitev (Last Updated):** Datum zadnje posodobitve (besedilo).
    - **Starostna omejitev (Content Rating):** Primernost vsebine glede na starost (npr. Za vse, Mature).
    - **Vsebuje oglase (Ad Supported):** Ali aplikacija vsebuje oglase (True/False).
    - **Nakupe v aplikaciji (In App Purchases):** Ali aplikacija omogoča nakupe v aplikaciji (True/False).
    - **Čas zajema podatkov (Scraped Time):** Datum in čas, ko so bili podatki zbrani.
    """)

# --- Category Counts ---
elif section == "Category Counts, Ratings & Installs":
    st.header("Category Usage Counts")
    st.markdown("""
    Ti grafi prikazujejo število aplikacij v posameznih kategorijah, kar nam omogoča vpogled v priljubljenost različnih kategorij aplikacij.
    Kategorije z večjim številom aplikacij so običajno bolj priljubljene med uporabniki, kar lahko nakazuje na večjo konkurenco in potencialno večje možnosti za monetizacijo.
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Most Used Categories")
        labels1 = category_counts.index[:10]
        values1 = category_counts.values[:10]
        fig, ax = plt.subplots()
        ax.bar(labels1, values1, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels1, rotation=45, ha="right")
        ax.set_ylabel("Number of Apps")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with col2:
        st.subheader("Bottom 10 Least Used Categories")
        labels2 = category_counts.index[-10:][::-1]
        values2 = category_counts.values[-10:][::-1]
        fig, ax = plt.subplots()
        ax.bar(labels2, values2, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels2, rotation=45, ha="right")
        ax.set_ylabel("Number of Apps")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.header("Category Ratings")
    st.markdown("""
    Ti grafi prikazujejo povprečne ocene aplikacij v posameznih kategorijah, kar nam omogoča vpogled v kakovost aplikacij glede na kategorijo.
    Kategorije z višjimi povprečnimi ocenami so običajno bolj priljubljene med uporabniki, kar lahko nakazuje na boljšo kakovost aplikacij v teh kategorijah.
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Best Rated Categories")
        labels1, values1 = zip(*sorted(category_avg_rating_list, key=lambda x: x[1], reverse=True)[:10])
        fig, ax = plt.subplots()
        ax.bar(labels1, values1, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels1, rotation=45, ha="right")
        ax.set_ylabel("Average Rating")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        min_y = min(values1)
        max_y = max(values1)
        margin = (max_y - min_y) * 0.15 if max_y != min_y else 0.1
        ax.set_ylim(min_y - margin, max_y + margin)
        yticks = np.arange(round(min_y - margin, 2), round(max_y + margin, 2), 0.1)
        ax.set_yticks(yticks)
        st.pyplot(fig)

    with col2:
        st.subheader("Bottom 10 Worst Rated Categories")
        labels2, values2 = zip(*sorted(category_avg_rating_list, key=lambda x: x[1])[:10])
        fig, ax = plt.subplots()
        ax.bar(labels2, values2, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels2, rotation=45, ha="right")
        ax.set_ylabel("Average Rating")
        ax.set_ylim(2.5, 3.1)
        ax.set_yticks(np.arange(2.5, 3.1, 0.1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.header("Category Installs")
    st.markdown("""
    Ti grafi prikazujejo povprečno število namestitev aplikacij v posameznih kategorijah, kar nam omogoča vpogled v priljubljenost aplikacij glede na število namestitev.
    Kategorije z višjim povprečnim številom namestitev so običajno bolj priljubljene med uporabniki, kar lahko nakazuje na večjo uporabniško bazo in potencialno večji dobiček za razvijalce.
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Most Installed Categories")
        labels1, values1 = zip(*sorted(category_avg_installs_list, key=lambda x: x[1], reverse=True)[:10])
        fig, ax = plt.subplots()
        ax.bar(labels1, values1, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels1, rotation=45, ha="right")
        ax.set_ylabel("Average Installs")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with col2:
        st.subheader("Bottom 10 Least Installed Categories")
        labels2, values2 = zip(*sorted(category_avg_installs_list, key=lambda x: x[1])[:10])
        fig, ax = plt.subplots()
        ax.bar(labels2, values2, color="royalblue", edgecolor="black")
        ax.set_xticklabels(labels2, rotation=45, ha="right")
        ax.set_ylabel("Average Installs")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig) 

# --- Monetization Types ---
elif section == "Monetization Types":
    st.header("Načini monetizacije aplikacij")
    st.markdown("""
    Ta razdelek analizira različne monetizacijske strategije aplikacij v Google Play Store, vključno s popolnoma brezplačnimi aplikacijami, aplikacijami z oglasi, plačljivimi aplikacijami in aplikacijami z nakupi v aplikaciji (IAP).
    """)
    labels = list(rating_price.keys())
    values = [len(rating_price[key]) for key in labels]
    colors = ['skyblue', 'lightgreen', 'gold', 'salmon', 'mediumpurple']
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'black'}
    )
    ax.legend(wedges, labels, title="App Types", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Distribution of App Monetization Types", fontsize=14)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Povprečne ocene aplikacij glede na načine monetizacije")
    st.markdown("""
    Ta graf prikazuje povprečne ocene aplikacij glede na različne načine monetizacije, kar nam omogoča vpogled v kakovost aplikacij glede na njihov monetizacijski model.
    Aplikacije z višjimi povprečnimi ocenami so običajno bolj priljubljene med uporabniki, kar lahko nakazuje na boljšo uporabniško izkušnjo in večjo vrednost za uporabnike.
    """)
    avg_ratings = [round(sum(ratings) / len(ratings), 2) if ratings else 0 for ratings in rating_price.values()]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, avg_ratings, color=colors, edgecolor='black')
    for bar, rating in zip(bars, avg_ratings):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rating:.2f}', ha='center', fontsize=10)
    ax.set_title("Average Ratings by Monetization Type", fontsize=14)
    ax.set_ylabel("Average Rating")
    ax.set_ylim(3, 5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)

# --- Bias in Categories ---
elif section == "Biases in Categories & Monetization":
    st.markdown("""
    ## Bias Analysis in Categories and Monetization Types
    Ta razdelek analizira naklonenosti v kategorijah aplikacij in načinih monetizacije, pri čemer se osredotoča na aplikacije z izstopajočimi lastnostmi glede na število namestitev in ocene.
    Odlične aplikacije so definirane kot aplikacije, ki bistveno presegajo povprečje v svoji kategoriji ali monetizacijskem tipu.
    """)
    st.subheader("Legenda grafov")
    st.markdown("""
    - **Bias Ratio**  
    Kolikokrat boljše so *izstopajoče aplikacije* v primerjavi s *preostalimi aplikacijami* glede na število namestitev.

    - **Standout Apps**  
    Aplikacije z **izjemno nadpovprečnim** številom namestitev.

    - **Remaining Apps**  
    Aplikacije s **okoli povprečnim** ali **podpovprečnim** številom namestitev.

    - **Total Installs (Standout)**  
    Vsota vseh namestitev za izstopajoče aplikacije.

    - **Total Installs (Remaining)**  
    Vsota vseh namestitev za preostale aplikacije.
    """)

    st.subheader("Bias in Categories (Installs & Ratings)")

    category_MaxInstall_all_id = (
        store[store['Installs'].notna()]
        .sort_values(['Category', 'Installs'], ascending=[True, False])
        .groupby('Category')
        .agg({
            'App Name': list,
            'App Id': list,
            'Installs': list
        })
        .apply(lambda row: list(zip(row['App Name'], row['App Id'], row['Installs'])), axis=1)
        .to_dict()
    )
    bias_multiplier = 2.5
    min_apps_required = 5
    ratios_installs_catagories = []
    top_avg_all_avg_catagories = {}
    for category, avg_install in category_avg_installs_list:
        apps = category_MaxInstall_all_id.get(category, [])
        if len(apps) < min_apps_required:
            continue
        sorted_apps = sorted(apps, key=lambda x: x[2], reverse=True)
        installs = [app[2] for app in sorted_apps]
        ids = [app[1] for app in sorted_apps]
        avg_all_installs = sum(installs) / len(installs)
        standout_apps = [app for app in sorted_apps if app[2] > bias_multiplier * avg_all_installs]
        if not standout_apps:
            continue
        standout_ids = [app[1] for app in standout_apps]
        standout_installs = [app[2] for app in standout_apps]
        standout_ratings = store.loc[store['App Id'].isin(standout_ids), 'Rating'].dropna().tolist()
        avg_top_rating = round(sum(standout_ratings) / len(standout_ratings), 2) if standout_ratings else "n/a"
        avg_top_installs = round(sum(standout_installs) / len(standout_installs))
        total_top_installs = sum(standout_installs)
        remaining_apps = [app for app in sorted_apps if app[1] not in standout_ids]
        if not remaining_apps:
            continue
        remaining_ids = [app[1] for app in remaining_apps]
        remaining_installs = [app[2] for app in remaining_apps]
        remaining_ratings = store.loc[store['App Id'].isin(remaining_ids), 'Rating'].dropna()
        avg_remaining_rating = round(remaining_ratings.mean(), 2) if not remaining_ratings.empty else "n/a"
        avg_remaining_installs = round(sum(remaining_installs) / len(remaining_installs))
        total_remaining_installs = sum(remaining_installs)
        Remaining_ratio = round(len(remaining_apps) / (len(standout_apps) + len(remaining_apps)) * 100, 1)
        bias_ratio = avg_top_installs / avg_remaining_installs if avg_remaining_installs > 0 else 0
        top_avg_all_avg_catagories[category] = [bias_ratio, Remaining_ratio, len(standout_apps), len(remaining_apps),
                                           avg_top_rating, avg_remaining_rating, total_top_installs, total_remaining_installs]
        ratios_installs_catagories.append((category, bias_ratio))
    columns = [
        "Bias Ratio", "Avg Ratio (%)", "Standout Apps", "Remaining Apps",
        "Avg Rating (Standout)", "Avg Rating (Remaining)",
        "Total Installs (Standout)", "Total Installs (Remaining)"
    ]
    df = pd.DataFrame.from_dict(
        top_avg_all_avg_catagories, orient='index', columns=columns
    )
    df_numeric = df.select_dtypes(include=['number']).drop(columns=["Avg Ratio (%)"])
    log_columns = [
        "Bias Ratio", "Standout Apps", "Remaining Apps",
        "Total Installs (Standout)", "Total Installs (Remaining)"
    ]
    linear_columns = ["Avg Rating (Standout)", "Avg Rating (Remaining)"]
    df_log = np.log10(df_numeric[log_columns] + 1e-3)
    df_linear = df_numeric[linear_columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 16), gridspec_kw={'width_ratios': [len(log_columns), len(linear_columns)]})
    sns.heatmap(df_log, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, ax=axes[0], cbar=True)
    axes[0].set_title("Log Scale")
    axes[0].set_ylabel("Category")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, ha='right')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    sns.heatmap(df_linear, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=axes[1], cbar=True)
    axes[1].set_title("Linear Scale")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Bias in Monetization Types (Installs & Ratings)")
    pricing_groups = {
        'Fully Free': filtered[fully_free],
        'With Ads': filtered[with_ads],
        'Paid': filtered[is_paid],
        'With IAP': filtered[with_iap],
        'With Ads + IAP': filtered[with_both]
    }
    bias_multiplier = 2.5
    min_apps_required = 5
    top_avg_all_avg_pricing = {}
    for pricing_method, df_group in pricing_groups.items():
        if len(df_group) < min_apps_required:
            continue
        df_sorted = df_group.sort_values('Installs', ascending=False)
        avg_all_installs = df_sorted['Installs'].mean()
        standout_apps = df_sorted[df_sorted['Installs'] > bias_multiplier * avg_all_installs]
        if standout_apps.empty:
            continue
        remaining_apps = df_sorted[~df_sorted['App Id'].isin(standout_apps['App Id'])]
        if remaining_apps.empty:
            continue
        standout_ratings = standout_apps['Rating'].dropna()
        remaining_ratings = remaining_apps['Rating'].dropna()
        avg_top_rating = round(standout_ratings.mean(), 2) if not standout_ratings.empty else "n/a"
        avg_remaining_rating = round(remaining_ratings.mean(), 2) if not remaining_ratings.empty else "n/a"
        standout_installs = standout_apps['Installs']
        remaining_installs = remaining_apps['Installs']
        avg_top_installs = round(standout_installs.mean())
        avg_remaining_installs = round(remaining_installs.mean())
        total_top_installs = standout_installs.sum()
        total_remaining_installs = remaining_installs.sum()
        total_apps = len(standout_apps) + len(remaining_apps)
        Remaining_ratio = round(len(remaining_apps) / total_apps * 100, 1)
        bias_ratio = avg_top_installs / avg_remaining_installs if avg_remaining_installs > 0 else 0
        top_avg_all_avg_pricing[pricing_method] = [bias_ratio, Remaining_ratio, len(standout_apps), len(remaining_apps),
                                           avg_top_rating, avg_remaining_rating, total_top_installs, total_remaining_installs]
    columns = [
        "Bias Ratio", "Avg Ratio (%)", "Standout Apps", "Remaining Apps",
        "Avg Rating (Standout)", "Avg Rating (Remaining)",
        "Total Installs (Standout)", "Total Installs (Remaining)"
    ]
    df = pd.DataFrame.from_dict(
        top_avg_all_avg_pricing, orient='index', columns=columns
    )
    df_numeric = df.select_dtypes(include=['number']).drop(columns=["Avg Ratio (%)"])
    log_columns = [
        "Bias Ratio", "Standout Apps", "Remaining Apps",
        "Total Installs (Standout)", "Total Installs (Remaining)"
    ]
    linear_columns = ["Avg Rating (Standout)", "Avg Rating (Remaining)"]
    df_log = np.log10(df_numeric[log_columns] + 1e-3)
    df_linear = df_numeric[linear_columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [len(log_columns), len(linear_columns)]})
    sns.heatmap(df_log, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, ax=axes[0], cbar=True)
    axes[0].set_title("Log Scale")
    axes[0].set_ylabel("Monetization Type")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, ha='right')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    sns.heatmap(df_linear, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=axes[1], cbar=True)
    axes[1].set_title("Linear Scale")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)   

# --- Model ---
elif section == "Prediction Model":

    st.markdown("""
    ## Napovedovalni Modela

    V tem razdelku lahko interaktivno napoveš oceno ali število namestitev aplikacije na podlagi izbranih lastnosti.
    Uporabnik določi kategorijo aplikacije, ali je brezplačna, ali vsebuje oglase, ali omogoča nakupe znotraj aplikacije ter število ocen.
    Na podlagi teh podatkov bo model napovedal predvideno oceno aplikacije ali predvideno skupno število njenih namestitev – glede na uporabnikovo izbiro.
    """)

    if 'target_variable' not in st.session_state:
        st.session_state['target_variable'] = 'Rating'
    if 'final_model' not in st.session_state:
        st.session_state['final_model'] = None
    if 'feature_columns' not in st.session_state:
        st.session_state['feature_columns'] = None

    target_variable = st.radio(
        "Izberi, kaj želiš napovedovati:",
        ['Rating', 'Installs'],
        index=0 if st.session_state['target_variable'] == 'Rating' else 1
    )

    if target_variable != st.session_state['target_variable'] or st.session_state['final_model'] is None:
        required_columns = ['Category', 'Free', 'Ad Supported', 'In App Purchases', 'Rating Count', target_variable]
        df = store.dropna(subset=required_columns)

        X = df[['Category', 'Free', 'Ad Supported', 'In App Purchases', 'Rating Count']]
        X = pd.get_dummies(X, columns=['Category'], drop_first=True)
        y = df[target_variable]

        model_path = f'final_random_forest_model_{target_variable.lower()}.pkl'
        if os.path.exists(model_path):
            final_model = joblib.load(model_path)
            st.info(f"Naložen model za napovedovanje **{target_variable}**.")
        else:
            final_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
            final_model.fit(X, y)
            joblib.dump(final_model, model_path)
            st.success(f"Model za napovedovanje **{target_variable}** je bil naučen in shranjen.")

        st.session_state['final_model'] = final_model
        st.session_state['feature_columns'] = X.columns
        st.session_state['target_variable'] = target_variable

    final_model = st.session_state['final_model']
    feature_columns = st.session_state['feature_columns']

    category_options = sorted(store['Category'].dropna().unique())

    with st.form("prediction_form"):
        category = st.selectbox("Kategorija", category_options)
        st.write("Lastnosti aplikacije:")
        is_free = st.checkbox("Brezplačna", value=True)
        has_ads = st.checkbox("Vsebuje oglase", value=True)
        has_iap = st.checkbox("Nakupe v aplikaciji", value=False)
        rating_count = st.number_input("Število ocen", min_value=0, step=100, value=1000)

        submitted = st.form_submit_button(f"Napovej {target_variable}")

    if submitted:
        input_data = {
            'Category': category,
            'Free': is_free,
            'Ad Supported': has_ads,
            'In App Purchases': has_iap,
            'Rating Count': rating_count
        }

        def predict_value(model, input_dict, feature_columns):
            df_input = pd.DataFrame([input_dict])
            df_encoded = pd.get_dummies(df_input, columns=['Category'], drop_first=True)
            df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
            return model.predict(df_encoded)[0]

        prediction = predict_value(final_model, input_data, feature_columns)
        st.subheader(f"Napovedana vrednost za {target_variable}:")
        if target_variable == 'Installs':
            st.write(f"**Predvideno: {int(prediction):,} namestitev**")
        else:
            st.write(f"**Predvidena ocena: {prediction:.2f}**")

# --- Old & new dataset comparison ---
elif section == "Primerjava starih in novih podatkov":
    st.markdown("""
    ## Primerjava starih in novih podatkov
    V tem razdelku primerjamo stare in nove podatke iz Google Play Store, da ugotovimo spremembe v povprečnem številu namestitev in ocen aplikacij po kategorijah.
    Uporabljamo podatke iz dveh različnih časovnih obdobij, da analiziramo trende in spremembe v priljubljenosti aplikacij.
    """)

    st.subheader("Legenda grafov")
    st.markdown("""
    - **Old Data**  
    Podatki so bili zbrani v letu 2021.

    - **New Data**  
    Podatki so bili zbrani v letu 2025.
    """)

    old_count = len(old_store)
    new_count = len(store)

    data_sizes = {
        'Old Dataset': old_count,
        'New Dataset': new_count
    }

    def thousands_formatter(x, pos):
        return f'{int(x/1000)}k' if x >= 1000 else str(int(x))

    st.markdown("""
                ### Graf skupnega števila aplikacij v starem in novem naboru podatkov
    Ta graf prikazuje skupno število aplikacij v starem in novem naboru podatkov. 
    Primerjava nam omogoča, da vidimo, kako se je število aplikacij spremenilo skozi čas.
            """)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(data_sizes.keys(), data_sizes.values(), color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Total Number of Apps: Old vs New Dataset')
    ax.set_ylabel('Number of Apps')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(bars, data_sizes.keys())
    plt.tight_layout()
    st.pyplot(fig)

    old_store['Maximum Installs'] = pd.to_numeric(old_store['Maximum Installs'], errors='coerce')
    store['Installs'] = pd.to_numeric(store['Installs'], errors='coerce')

    combined_installs = pd.concat([old_store['Maximum Installs'], store['Installs']])

    min_installs = int(np.floor(combined_installs.min()))
    max_installs = int(np.ceil(combined_installs.max()))

    log_bins = np.logspace(np.log10(min_installs), np.log10(max_installs), num=11)

    def round_to_base(x, base=10000):
        return int(base * round(x / base))

    rounded_edges = [round_to_base(x) for x in log_bins]
    rounded_edges = sorted(set(rounded_edges))

    labels = []
    for i in range(len(rounded_edges) - 1):
        start = rounded_edges[i]
        end = rounded_edges[i+1] - 1
        labels.append(f"{start:,} - {end:,}")

    old_bins = pd.cut(old_store['Maximum Installs'], bins=log_bins, labels=labels, include_lowest=True)
    new_bins = pd.cut(store['Installs'], bins=log_bins, labels=labels, include_lowest=True)

    old_counts = old_bins.value_counts().sort_index()
    new_counts = new_bins.value_counts().sort_index()

    df_bins = pd.DataFrame({
        "Old Data": old_counts,
        "New Data": new_counts
    })

    st.markdown("""
                ### Graf porazdelitve aplikacij glede na število namestitev
    Ta graf prikazuje porazdelitev aplikacij glede na število namestitev v starem in novem naboru podatkov.
    Uporabljamo logaritemske razrede, da bolje prikažemo širok razpon števila namestitev.
    """)

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    df_bins.plot(kind='bar', ax=ax2, width=0.85)
    ax2.set_title("App Distribution by Install Ranges")
    ax2.set_xlabel("Install Range")
    ax2.set_ylabel("Number of Apps")
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_yscale('log')
    st.pyplot(fig2)

    st.markdown("""
                ### Graf primerjave povprečnega števila namestitev in ocen aplikacij po kategorijah
    Ta graf prikazuje primerjavo povprečnega števila namestitev in ocen aplikacij po kategorijah med starimi in novimi podatki.
            """)

    old_store['Maximum Installs'] = pd.to_numeric(old_store['Maximum Installs'], errors='coerce')
    old_category_avg_installs = (
        old_store[old_store['Maximum Installs'].notna()]
        .groupby('Category')['Maximum Installs']
        .mean()
        .round(0)
        .astype(int)
        .to_dict()
    )
    new_category_avg_installs = (
        store[store['Installs'].notna()]
        .groupby('Category')['Installs']
        .mean()
        .round(0)
        .astype(int)
        .to_dict()
    )

    old_category_avg_rating = (
        old_store[old_store['Rating'].notna()]
        .groupby('Category')['Rating']
        .mean()
        .round(2)
        .to_dict()
    )
    new_category_avg_rating = (
        store[store['Rating'].notna()]
        .groupby('Category')['Rating']
        .mean()
        .round(2)
        .to_dict()
    )

    categories = set(old_category_avg_installs) | set(new_category_avg_installs)
    df_compare = pd.DataFrame({
        "Old Avg Installs": pd.Series(old_category_avg_installs),
        "New Avg Installs": pd.Series(new_category_avg_installs),
        "Old Avg Rating": pd.Series(old_category_avg_rating),
        "New Avg Rating": pd.Series(new_category_avg_rating)
    })
    df_compare["Avg Installs Change"] = df_compare["New Avg Installs"] - df_compare["Old Avg Installs"]
    df_compare["Avg Rating Change"] = df_compare["New Avg Rating"] - df_compare["Old Avg Rating"] 

    # --- Installs comparison ---
    top_n = 48
    top_installs = df_compare.dropna().sort_values('Avg Installs Change', ascending=False).head(top_n)

    categories = top_installs.index
    old_vals = top_installs['Old Avg Installs']
    new_vals = top_installs['New Avg Installs']

    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, old_vals, marker='o', linestyle='-', color='skyblue', label='Old Avg Installs')
    ax.plot(x, new_vals, marker='o', linestyle='-', color='orange', label='New Avg Installs')

    ax.set_ylabel('Average Installs')
    ax.set_title(f'Categories by Change in Avg Installs')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Graf primerjave povprečnih ocen aplikacij po kategorijah med starimi in novimi podatki
    Ta graf prikazuje primerjavo povprečnih ocen aplikacij po kategorijah med starimi in novimi podatki.
    Spremembe v povprečnih ocenah lahko kažejo na izboljšave ali poslabšanja kakovosti aplikacij v posameznih kategorijah.    
    """)

    # --- Ratings comparison ---
    top_n = 48
    top_ratings = df_compare.dropna().sort_values('Avg Rating Change', ascending=False).head(top_n)

    categories = top_ratings.index
    old_vals = top_ratings['Old Avg Rating']
    new_vals = top_ratings['New Avg Rating']

    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, old_vals, marker='o', linestyle='-', color='skyblue', label='Old Avg Rating')
    ax.plot(x, new_vals, marker='o', linestyle='-', color='orange', label='New Avg Rating')

    ax.set_ylabel('Average Rating')
    ax.set_title(f'Categories by Change in Avg Rating')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    all_vals = np.concatenate([old_vals, new_vals])
    y_min = max(0, all_vals.min() - 0.1)
    y_max = min(5, all_vals.max() + 0.1)
    ax.set_ylim(y_min, y_max)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Graf primerjave povprečnega števila namestitev in ocen aplikacij po načinih monetizacije med starimi in novimi podatki
    Ta graf prikazuje primerjavo povprečnega števila namestitev in ocen aplikacij po načinih monetizacije med starimi in novimi podatki.
    Spremembe v povprečnem številu namestitev in ocenah lahko kažejo na trende v priljubljenosti aplikacij glede na njihove monetizacijske modele.      
    """)

    old_store['Maximum Installs'] = pd.to_numeric(old_store['Maximum Installs'], errors='coerce')

    filtered_old = old_store[(old_store['Rating'].notna()) & (old_store['Rating'] != 0.0)]
    filtered_old = filtered_old[filtered_old['Rating'] < 5.0]

    is_paid_old = ~filtered_old['Free']
    has_ads_old = filtered_old['Ad Supported'] == True
    has_iap_old = filtered_old['In App Purchases'] == True
    fully_free_old = filtered_old['Free'] & ~has_ads_old & ~has_iap_old
    with_ads_old = filtered_old['Free'] & has_ads_old & ~has_iap_old
    with_iap_old = filtered_old['Free'] & ~has_ads_old & has_iap_old
    with_both_old = filtered_old['Free'] & has_ads_old & has_iap_old

    pricing_groups_old = {
        'Fully Free': filtered_old[fully_free_old],
        'With Ads': filtered_old[with_ads_old],
        'Paid': filtered_old[is_paid_old],
        'With IAP': filtered_old[with_iap_old],
        'With Ads + IAP': filtered_old[with_both_old]
    }

    pricing_groups = {
        'Fully Free': filtered[fully_free],
        'With Ads': filtered[with_ads],
        'Paid': filtered[is_paid],
        'With IAP': filtered[with_iap],
        'With Ads + IAP': filtered[with_both]
    }

    bias_multiplier = 2.5
    min_apps_required = 5

    top_avg_all_avg_pricing = compute_pricing_bias(
    pricing_groups,
    installs_col='Installs',
    rating_col='Rating',
    app_id_col='App Id',
    bias_multiplier=bias_multiplier,
    min_apps_required=min_apps_required
    )
    
    
    top_avg_all_avg_pricing_old = {}

    for pricing_method, df in pricing_groups_old.items():
        if len(df) < min_apps_required:
            continue

        df_sorted = df.sort_values('Maximum Installs', ascending=False)
        avg_all_installs = df_sorted['Maximum Installs'].mean()

        standout_apps = df_sorted[df_sorted['Maximum Installs'] > bias_multiplier * avg_all_installs]
        if standout_apps.empty:
            continue

        remaining_apps = df_sorted[~df_sorted['App Id'].isin(standout_apps['App Id'])]
        if remaining_apps.empty:
            continue

        standout_ratings = standout_apps['Rating'].dropna()
        remaining_ratings = remaining_apps['Rating'].dropna()
        avg_top_rating = round(standout_ratings.mean(), 2) if not standout_ratings.empty else "n/a"
        avg_remaining_rating = round(remaining_ratings.mean(), 2) if not remaining_ratings.empty else "n/a"

        standout_installs = standout_apps['Maximum Installs']
        remaining_installs = remaining_apps['Maximum Installs']
        avg_top_installs = round(standout_installs.mean())
        avg_remaining_installs = round(remaining_installs.mean())
        total_top_installs = standout_installs.sum()
        total_remaining_installs = remaining_installs.sum()

        total_apps = len(standout_apps) + len(remaining_apps)
        Remaining_ratio = round(len(remaining_apps) / total_apps * 100, 1)
        bias_ratio = avg_top_installs / avg_remaining_installs if avg_remaining_installs > 0 else 0

        top_avg_all_avg_pricing_old[pricing_method] = [
            bias_ratio, Remaining_ratio, len(standout_apps), len(remaining_apps),
            avg_top_rating, avg_remaining_rating, total_top_installs, total_remaining_installs
        ]

    df_new_pricing = pd.DataFrame.from_dict(top_avg_all_avg_pricing, orient='index',
        columns=['Bias Ratio New', 'Remaining % New', 'Standout Count New', 'Remaining Count New',
                'Avg Top Rating New', 'Avg Remaining Rating New', 'Total Top Installs New', 'Total Remaining Installs New'])

    df_old_pricing = pd.DataFrame.from_dict(top_avg_all_avg_pricing_old, orient='index',
        columns=['Bias Ratio Old', 'Remaining % Old', 'Standout Count Old', 'Remaining Count Old',
                'Avg Top Rating Old', 'Avg Remaining Rating Old', 'Total Top Installs Old', 'Total Remaining Installs Old'])

    df_pricing_compare = df_old_pricing.join(df_new_pricing, how='outer').fillna(0)

    df_pricing_compare['Bias Ratio Change'] = df_pricing_compare['Bias Ratio New'] - df_pricing_compare['Bias Ratio Old']
    df_pricing_compare['Avg Top Rating Change'] = df_pricing_compare['Avg Top Rating New'] - df_pricing_compare['Avg Top Rating Old']
    df_pricing_compare['Avg Remaining Rating Change'] = df_pricing_compare['Avg Remaining Rating New'] - df_pricing_compare['Avg Remaining Rating Old']
    df_pricing_compare['Standout Count Change'] = df_pricing_compare['Standout Count New'] - df_pricing_compare['Standout Count Old']

    sns.set(style='whitegrid')
    columns = [
        "Bias Ratio Old", "Standout Count Old", "Remaining Count Old",
        "Avg Top Rating Old", "Avg Remaining Rating Old",
        "Total Top Installs Old", "Total Remaining Installs Old",
        "Bias Ratio New", "Standout Count New", "Remaining Count New",
        "Avg Top Rating New", "Avg Remaining Rating New",
        "Total Top Installs New", "Total Remaining Installs New"
    ]

    df_heat = df_pricing_compare[columns]

    log_columns = [
        "Bias Ratio Old", "Standout Count Old", "Remaining Count Old",
        "Total Top Installs Old", "Total Remaining Installs Old",
        "Bias Ratio New", "Standout Count New", "Remaining Count New",
        "Total Top Installs New", "Total Remaining Installs New"
    ]
    linear_columns = [
        "Avg Top Rating Old", "Avg Remaining Rating Old",
        "Avg Top Rating New", "Avg Remaining Rating New"
    ]

    df_log = df_heat[log_columns]
    df_log = np.log10(df_log + 1e-3)

    df_linear = df_heat[linear_columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 12), gridspec_kw={'width_ratios': [len(log_columns), len(linear_columns)]})

    sns.heatmap(
        df_log,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=axes[0],
        cbar=True
    )
    axes[0].set_title("Log Scale Metrics (Counts & Bias Ratios)")
    axes[0].set_ylabel("Pricing Group")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, ha='right')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(
        df_linear,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title("Linear Scale Metrics (Ratings)")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([]) 
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Graf primerjave povprečnega števila namestitev in ocen aplikacij po kategorijah med starimi in novimi podatki
    Ta graf prikazuje primerjavo povprečnega števila namestitev in ocen aplikacij po kategorijah med starimi in novimi podatki.
    Spremembe v povprečnem številu namestitev in ocenah lahko kažejo na trende v priljubljenosti aplikacij v posameznih kategorijah.       
    """)

    bias_multiplier = 2.5
    min_apps_required = 5

    def compute_category_bias(store_df, installs_col, rating_col, app_id_col):
        store_df[installs_col] = pd.to_numeric(store_df[installs_col], errors='coerce')
        filtered = store_df[(store_df[rating_col].notna()) & (store_df[rating_col] < 5.0)]

        categories = filtered['Category'].unique()
        category_bias_stats = {}

        for category in categories:
            cat_apps = filtered[filtered['Category'] == category]
            if len(cat_apps) < min_apps_required:
                continue

            cat_apps_sorted = cat_apps.sort_values(installs_col, ascending=False)
            avg_all_installs = cat_apps_sorted[installs_col].mean()

            # Identify standout apps
            standout_apps = cat_apps_sorted[cat_apps_sorted[installs_col] > bias_multiplier * avg_all_installs]
            if standout_apps.empty:
                continue

            remaining_apps = cat_apps_sorted[~cat_apps_sorted[app_id_col].isin(standout_apps[app_id_col])]
            if remaining_apps.empty:
                continue

            # Ratings
            standout_ratings = standout_apps[rating_col].dropna()
            remaining_ratings = remaining_apps[rating_col].dropna()

            avg_top_rating = round(standout_ratings.mean(), 2) if not standout_ratings.empty else np.nan
            avg_remaining_rating = round(remaining_ratings.mean(), 2) if not remaining_ratings.empty else np.nan

            # Installs
            standout_installs = standout_apps[installs_col]
            remaining_installs = remaining_apps[installs_col]

            avg_top_installs = round(standout_installs.mean())
            avg_remaining_installs = round(remaining_installs.mean())
            total_top_installs = standout_installs.sum()
            total_remaining_installs = remaining_installs.sum()

            total_apps = len(standout_apps) + len(remaining_apps)
            remaining_ratio = round(len(remaining_apps) / total_apps * 100, 1)
            bias_ratio = avg_top_installs / avg_remaining_installs if avg_remaining_installs > 0 else 0

            category_bias_stats[category] = [
                bias_ratio, remaining_ratio, len(standout_apps), len(remaining_apps),
                avg_top_rating, avg_remaining_rating, total_top_installs, total_remaining_installs
            ]

        return category_bias_stats


    old_category_bias = compute_category_bias(
        old_store,
        installs_col='Maximum Installs',
        rating_col='Rating',
        app_id_col='App Id'
    )

    new_category_bias = compute_category_bias(
        store,
        installs_col='Installs',
        rating_col='Rating',
        app_id_col='App Id'
    )

    columns = [
        "Bias Ratio", "Avg Ratio (%)", "Standout Apps", "Remaining Apps",
        "Avg Rating (Standout)", "Avg Rating (Remaining)",
        "Total Installs (Standout)", "Total Installs (Remaining)"
    ]

    df_old_cat_bias = pd.DataFrame.from_dict(old_category_bias, orient='index', columns=columns)
    df_new_cat_bias = pd.DataFrame.from_dict(new_category_bias, orient='index', columns=columns)

    df_cat_bias_compare = df_old_cat_bias.add_suffix(' Old').join(df_new_cat_bias.add_suffix(' New'), how='outer').fillna(0)

    df_cat_bias_compare['Bias Ratio Change'] = df_cat_bias_compare['Bias Ratio New'] - df_cat_bias_compare['Bias Ratio Old']
    df_cat_bias_compare['Avg Top Rating Change'] = df_cat_bias_compare['Avg Rating (Standout) New'] - df_cat_bias_compare['Avg Rating (Standout) Old']
    df_cat_bias_compare['Avg Remaining Rating Change'] = df_cat_bias_compare['Avg Rating (Remaining) New'] - df_cat_bias_compare['Avg Rating (Remaining) Old']
    df_cat_bias_compare['Standout Count Change'] = df_cat_bias_compare['Standout Apps New'] - df_cat_bias_compare['Standout Apps Old']

    df_cat_bias_compare.sort_values('Bias Ratio Change', ascending=False, inplace=True)


    # --- Heatmap for Category Bias Comparison ---
    log_columns = [
        "Bias Ratio Old", "Standout Apps Old", "Remaining Apps Old",
        "Total Installs (Standout) Old", "Total Installs (Remaining) Old",
        "Bias Ratio New", "Standout Apps New", "Remaining Apps New",
        "Total Installs (Standout) New", "Total Installs (Remaining) New"
    ]

    linear_columns = [
        "Avg Rating (Standout) Old", "Avg Rating (Remaining) Old",
        "Avg Rating (Standout) New", "Avg Rating (Remaining) New"
    ]

    df_log = df_cat_bias_compare[log_columns].fillna(0)
    df_linear = df_cat_bias_compare[linear_columns].fillna(0)

    df_log = np.log10(df_log + 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(18, 16), gridspec_kw={'width_ratios': [len(log_columns)//2, len(linear_columns)//2]})

    sns.heatmap(
        df_log,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=axes[0],
        cbar=True
    )
    axes[0].set_title("Log Scale Metrics (Bias, Counts, Installs)")
    axes[0].set_ylabel("Category")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, ha='right')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(
        df_linear,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title("Linear Scale Metrics (Ratings)")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([]) 
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)

# --- Raw Data ---
elif section == "Raw Data":
    st.markdown("""
    ## Raw Data
    Ta tabela prikazuje surove podatke iz Google Play Store, ki so bili uporabljeni v analizi.
    Vsebuje informacije o aplikacijah, vključno z imenom aplikacije, kategorijo, številom namestitev, ocenami in monetizacijskimi strategijami.
    """)
    st.header("Raw Data")
    st.dataframe(store)
    st.write(store.describe())

    st.markdown("### Poišči aplikacijo")

    search_text = st.text_input("Za pohitritev iskanja vnesi delno ali celo ime aplikacije",)

    min_chars = 2
    selected_app = None

    if len(search_text) >= min_chars:
        app_names = store['App Name'].dropna().unique()
        filtered_names = sorted([name for name in app_names if search_text.lower() in name.lower()])
        
        if filtered_names:
            selected_app = st.selectbox("Izberi aplikacijo", options=filtered_names)
        else:
            st.info("Ni zadetkov za vneseni iskalni niz.")

    elif search_text:
        st.warning("Vpiši vsaj 2 znaka za začetek iskanja.")

    if selected_app:
        app_row = store[store['App Name'] == selected_app]
        if not app_row.empty:
            st.write("**Podatki za izbrano aplikacijo:**")
            st.write(app_row.T)





