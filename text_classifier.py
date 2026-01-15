import nltk
import numpy as np
import pandas as pd

nltk.download("all")

dataset = pd.read_csv('https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Restaurant_Reviews.tsv.txt', delimiter= '\t', quoting = 3)
print("Here's the dataset head xxxxxx")

print(dataset.head())

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()
print("Here's the dataset info xxxxxxxxx")
print(dataset.info())

