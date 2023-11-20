import pandas as pd 
import numpy as np
from itertools import islice

"""
Propojení s účet Azure AI Services. Vyžaduje zadání language_key a language_endpoint
z vlastního účtu
"""
language_key = '4793e586419a47eaafffae7708470291'
language_endpoint = 'https://mynewrecource.cognitiveservices.azure.com'

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

"""
Autentifikace klienta s použitím key a endpoint.
"""
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

"""
Příprava DataFrame na další zpracování. Načtení DataFrame sentiment z csv. Nahrazení prázdných hodnot a odstranění řádků 
s chybějícími hodnotami. Vytváření sloupce 'itemUrl_HEUREKA', jako PK. Sloupec s PK
a reviews převádíme do slovníku. Následně vytváříme seznam slovníků.   
"""
sentiment = pd.read_csv('./Union_reviews.csv')
sentiment['reviews'].replace('',np.nan,inplace = True)
sentiment.dropna(subset=['reviews'], inplace=True)
sentiment['id'] = sentiment['itemUrl_HEUREKA']
sentiment['text'] = sentiment['reviews']
sentiment = sentiment[['id','text']]
sentiment = sentiment.to_dict('records') 
print(len(sentiment))

"""
Program zvládne zpracovat pouze 10 záznamů. Proto používáme funkci batch, která 
rozděluje objekt do dávek, které budeme pouštet do programu pro analýzu sentimentu.
"""

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

"""
Samotná analýza sentimentu. U které zaznamenáváme výsledný sentiment, 
hodnoty positive, neutral, negative a v DataFrame zachováváme id pro napojení
na tabulku s ratingem. 
"""

def get_sentiment(client):

    data = []
    i = 0
    for batch in batched(sentiment, 10):
        print('Analyzing #', i) 
        i = i + 1
        documents = batch
        result = client.analyze_sentiment(documents, show_opinion_mining=True)

        for item in result:
            if item.is_error:
                print('Error')
                continue

            data.append([item['id'],item['sentiment'],item['confidence_scores'].get('positive'),item['confidence_scores'].get('negative'),item['confidence_scores'].get('neutral')])

    return data

print(client)


