import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import  LatentDirichletAllocation as LDA
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import torch
# import transformers as ppb # pytorch transformers
# import swifter
import tqdm
# from transformers import DistilBertForSequenceClassification,DistilBertTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

import nltk
nltk.download('stopwords')
nltk.download('punkt') 

app = Flask(__name__)
model = pickle.load(open('mrsvm.pkl', 'rb'))

@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)



def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem    

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])


# processing the request from dialogflow
def processRequest(req):

    #sessionID=req.get('responseId')
    result = req.get("queryResult")
    #user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    review = parameters.get("movie_review")

    f1 = clean(review[0])
    f2 = is_special(f1)
    f3 = to_lower(f2)
    f4 = rem_stopwords(f3)
    f5 = stem_txt(f4)
    f5 =[f5]
       

    file = open("voc.txt", "r")

    contents = file.read()
    voc = ast.literal_eval(contents)
    file.close()

    # Creating a corpus
    corpus = f5
    X_tf = TfidfVectorizer(vocabulary=voc)
    X_tfd =  X_tf.fit_transform(corpus).toarray()
    X_tfd = pd.DataFrame(X_tfd)


    final_review = X_tfd
     
    intent = result.get("intent").get('displayName')
    
    if (intent=='Movie_Review'):
        prediction = model.predict(final_review)
    
        output = round(prediction[0], 1)
    
        
        if(output==0):
            sentiment = 'Negative'
    
        if(output==1):
            sentiment = 'Positive'
        
       
        fulfillmentText= "The Sentiment of your review is ..  {} !".format(sentiment)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    #else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)

if __name__ == '__main__':
    app.run()
#if __name__ == '__main__':
#    port = int(os.getenv('PORT', 5000))
#    print("Starting app on port %d" % port)
#    app.run(debug=False, port=port, host='0.0.0.0')