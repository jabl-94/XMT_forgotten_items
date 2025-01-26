import csv
import re
from all_cat_map import cod_mkt_cat2name


def translate_to_english(italian_text):
    # Common translations
    translations = {
        'alimentari': 'food',
        'non alimentari': 'non_food',
        'igiene': 'hygiene',
        'persona': 'personal',
        'prodotti': 'products',
        'sanitari': 'sanitary',
        'primo soccorso': 'first_aid',
        'parafarmaceutici': 'parapharmaceuticals',
        'automedicazione': 'self_medication',
        'casa': 'home',
        'cucina': 'kitchen',
        'bagno': 'bathroom',
        'pulizia': 'cleaning',
        'cura': 'care',
        'abbigliamento': 'clothing',
        'accessori': 'accessories',
        'bevande': 'beverages',
        'frutta': 'fruit',
        'verdura': 'vegetables',
        'carne': 'meat',
        'pesce': 'fish',
        'pane': 'bread',
        'pasta': 'pasta',
        'latticini': 'dairy',
        'uova': 'eggs',
        'dolci': 'sweets',
        'snack': 'snacks',
        'surgelati': 'frozen_food',
        'conserve': 'preserves',
        'condimenti': 'condiments',
        'elettronica': 'electronics',
        'giardinaggio': 'gardening',
        'animali': 'pets',
        'cancelleria': 'stationery',
        'giocattoli': 'toys',
        'sport': 'sports',
        'auto': 'car',
        'moto': 'motorcycle',
        'biciclette': 'bicycles',
    }

    # Convert to lowercase and split into words
    words = italian_text.lower().split()

    # Translate each word if it's in our dictionary, otherwise keep it as is
    translated_words = [translations.get(word, word) for word in words]

    # Join the words and remove any non-alphanumeric characters
    english_text = '_'.join(translated_words)
    english_text = re.sub(r'[^a-zA-Z0-9_]', '', english_text)

    return english_text


def generate_cat_map():
    new_cat_map = {}

    # Read market.csv and process each row
    with open('market.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            prefix = row['cod_mkt'][:7]
            categoria = translate_to_english(row['categoria'])

            if prefix in cod_mkt_cat2name:
                new_cat_map[prefix] = translate_to_english(cod_mkt_cat2name[prefix])
            else:
                new_cat_map[prefix] = categoria

    # Write the new dictionary to cat_map.py
    with open('../../Experiments/cat_map.py', 'w', encoding='utf-8') as f:
        f.write("cod_mkt_cat2name = {\n")
        for k, v in new_cat_map.items():
            f.write(f"    \"{k}\": \"{v}\",\n")
        f.write("}\n")


if __name__ == "__main__":
    generate_cat_map()