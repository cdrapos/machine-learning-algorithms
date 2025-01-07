#Reading CSV file
import chardet
with open(r'C:\Users\Desktop\yourcorpus.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print(result)
import pandas as pd
input_file = r'C:\Users\Desktop\yourcorpus2.csv'
output_file = 'output.csv'
input_encoding = 'utf-16'  # Replace with the actual encoding of your file, e.g., 'ISO-8859-1'
# Open the input file with the specified encoding and write it to a new file with UTF-8 encoding.
with open(input_file, 'r', encoding=input_encoding) as file:
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(file.read())
df = pd.read_csv(output_file)
print(df)

#Clear text
text='Enter the text you want to clean here'
text = text.lower()
# Remove punctuation marks
punctuation_marks = ['','ʼ','.','?','‡','-','“','=','&','$','€','§','°','_','·','᾿','|','#','%','€','$ ','∙','‘‘','‘','/','~','*','&','”','/','~','','κεφαλαιο','Κεφάλαιο','"','*','•','Σελίδα','•','','·','…','..','...','·','·','’','@','►','◄',',', '.','΄','΄', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}','—','', '«','»','―', ' ' ' ','΄',' ’ ' , ' · ' , '  " ' , ' ’ ' ,'–','0','1','2','3','4','5','6','7','8','9','+','☺','♠','®','☻','♥','æ','±']
for mark in punctuation_marks:
    text = text.replace(mark, '')
# Remove tones (assuming diacritics)
import unicodedata
text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
# Print pre-edited text
print(text)


#Introduction of stopwords in Greek
import nltk
from nltk.corpus import stopwords
sw_nltk = stopwords.words('greek')
print(sw_nltk)

#Remove terminal terms
text='Enter the text you want to clean here'
words = [word for word in text.split() if word.lower() not in sw_nltk]
new_text = " ".join(words)
print(new_text)
print("Old length: ", len(text))
print("New length: ", len(new_text))

#Discretion
text='Enter the text you want to clean here'
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
# word tokenization
print("Word Tokenization:")
print(word_tokenize(text))

#Staffing
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem
sentence = 'Enter a sentence here'
words = word_tokenize(sentence)
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed words:", stemmed_words)

#Limmatization
import spacy
nlp = spacy.load('el_core_news_md')
text='''Text Here'''
doc = nlp(text)
#for token in doc:
    #print("Original token: {} , Lemma: {}".format(token,token.lemma_))
    #print("{}".format(token.lemma_))
lemmas = " ".join(token.lemma_ for token in doc)
print(lemmas)
