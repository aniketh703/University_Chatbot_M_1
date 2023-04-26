import json
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf
import tflearn
from flask import Flask, make_response, render_template, request
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

stemmer = LancasterStemmer()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def model_predict(inp):

    with open("intents.json") as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    model = Sequential()
    model.add(Dense(units=8, input_dim=len(training[0]), activation='relu'))
    model.add(Dense(units=len(output[0]), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    global graph
    graph = tf.compat.v1.get_default_graph()

    try:
        model.load_weights("model.tflearn")
    except:
        model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
        model.save("model.tflearn")


    with graph.as_default():
        results = model.predict(np.array([bag_of_words(inp, words)]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if 'tag' in tg and tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)
        else:
            return "Not Understood!"

app= Flask(__name__)
@app.route('/')
def hello_world():
    return render_template("Home.html")
@app.route('/home')
def home():
    return render_template("Home.html")
@app.route('/contact')
def contact():
    return render_template("Contact.html")
@app.route('/about')
def about():
    return render_template("About.html")
@app.route('/services')
def services():
    return render_template("Services.html")

@app.route('/bot')
def bot():
    data=""
    disp=""
    return render_template("index.html", data=data)



@app.route('/form_return',methods=['POST','GET'])
def chat():
    print("start talking with the bot (type quit to stop)!")
    user1=request.form['user']
    inp = user1
    print("-----------------------------------------------------------")
    print(inp,type(inp))
    data = model_predict(inp)
    if data == "Not Understood!":
        return render_template('index.html', data=data, disp=inp)
    else:
        return render_template('index.html', data=data, disp=inp)
    # return render_template('index.html', data=data,disp=inp)
if __name__=='__main__':
    app.run()
app.run(debug=True)  