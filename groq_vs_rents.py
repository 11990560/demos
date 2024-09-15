!pip install groq

from google.colab import userdata
import os
import requests
import pandas as pd
from groq import Groq

# augmentation input
df = pd.read_csv('https://raw.githubusercontent.com/11990560/demos/main/test_df.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/11990560/demos/main/calgary_rents_raw.csv')
# print(df)

df = df.iloc[2500:3500]

# data = df.to_string()
data = round(df['psf'],1).to_string()
# data = round(df['price'],1).to_string()
# data = round(df[['price','sq_feet']],1).to_string()
# print(data)

print("input string length: ",len(data))

client = Groq(
    api_key=userdata.get('GROQ_API_KEY'),
    # api_key= 'GROQ_API_KEY',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            # "content": "you are a stressed out real estate genius."
            # "content": "you are a stressed out real estate genius that never hallucinates."
            # "content": "you are a very concise, yet accurate, stressed out real estate genius that never hallucinates."
            "content": "you are a very concise, yet accurate, real estate genius that never makes up, extrapolates, or arbitrarily fills in numbers, or other data."
        },
        {
            "role": "user",
            "content": "I am giving you a pandas dataframe of rental rates from Calgary.",
        },
        {
            "role": "user",
            "content": data,
        },
        {
            "role": "user",
            "content": "Give me some statistics about rentals in Calgary.",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)
