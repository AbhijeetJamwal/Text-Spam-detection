from django.shortcuts import render

# Create your views here.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #to convert text data to numerical data
from sklearn.model_selection import train_test_split  #to split data
from sklearn.naive_bayes import MultinomialNB       #NaiveBayes for text classification
from sklearn.metrics import accuracy_score  
from .forms import MessageForm

df = pd.read_csv("D:/Projects/Spam detection/emails.csv")

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df["text"])

#splitting data into training and testing 

x_train,x_test,y_train,y_test = train_test_split(x,df["spam"],test_size=0.2)

model = MultinomialNB() 
model = model.fit(x_train,y_train)


def predictMessage(message):
    messageVector = vectorizer.transform([message])
    prediction = model.predict(messageVector)
    return "spam" if prediction[0] == 1 else "ham"


def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predictMessage(message)
    else:
        form = MessageForm()
    return render(request,'home.html',{'form' : form , 'result' : result})


