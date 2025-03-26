# Podatkovno rudarjenje Google Play Store aplikacij
### Skupina 11

|Ime|
|:------- |  
|Filip Vrabec|
|Klemen Krkovič |
|Lan Miglič|

## Opis problema 

- Kaj so najbolj popularne aplikacije (po ocenah, po namestitvah).
- Kaj je najbolj pogosta velikost aplikacij.
- Katera kategorija ima največ aplikacij.
- Popularnosti in velikosti po kategorijah.
- V katere kategorije spadajo top 10 najbolj popularnih aplikacij, top 100…
- Razlika o popularnosti (po ocenah, po namestitvah) med brezplačnimi in plačljivimi aplikacijami

## Podatki

Vir: https://www.kaggle.com/datasets/lava18/google-play-store-apps/data


**ATRIBUTI:**
* `app` (Ime aplikacije) (*string*)
* `category` (Kategorija, kateri aplikacija pripada) (*string*)
* `rating`  (Skupna ocena aplikacije) (*float*)
* `reviews` (Število ocen uporabnikov) (*int*)
* `size` (Velikost aplikacije) (*float*)
* `installs` (Število prenosov) (*int*)
* `type` (Je aplikacija brezplačna) ( *enum* { free | paid } )
* `price` (Cena aplikacije (*float*)
* `content rating` (Starostno priporočilo) ( *enum* { everyone | teen | mature | adult } )
* `genres` (Možne dodatne kategorije (vključno z glavno)) ([*strings*])
* `last updated` (Datum zadnje posodobitve) (*string*)
* `current version` (Trenutna različica aplikacije) (*string*)
* `android version` (Minimalna različica androida) (*string*)

