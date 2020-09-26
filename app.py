from flask import Flask, render_template, request,url_for
import nltk 
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import string
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle 
import joblib
import csv

clf = joblib.load(open("gnb.pkl",'rb'))
cv = joblib.load(open("cv_transform.pkl",'rb'))

stop_words = {'down', "don't", "shouldn't", 'out', 'both', '+', 'should', "wasn't", 'will', 'he', 'am', 'before', 'o', 'wasn', 'any', 'after', '\\', 'during', 'these', 'have', 'just', 'further', '{', 'weren', 'in', "shan't", 'does', 'so', 'into', 'herself', 'too', "'", 'she', 'doing', 'my', "it's", 'no', 'how', "won't", 'their', 'this', 'couldn', 'y', '%', "you've", 'i', 'her', 'again', 'hasn', 'ourselves', 'when', '?', 'was', '`', 've', 'of', 'over', 'from', 'but', 'same', ')', '|', 'ain', "you'll", 'himself', 'for', 'has', "hasn't", '!', '[', 'that', 'what', ';', 'only', 'll', 'd', 'your', 'those', 'being', 'nor', 'haven', "you'd", 'as', 'while', 'because', "couldn't", 't', 'which', 'and', '$', ',', '/', "she's", '<', '*', 'needn', 'we', '#', 'shouldn', 'is', 'mightn', 'hers', '(', ':', 'whom', 'be', '@', 'hadn', 'by', 'myself', 'ma', 'yourself', 'theirs', 'the', 'under', 'than', 'isn', "mightn't", 'aren', "haven't", 'an', ']', "didn't", "mustn't", 'once', 'with', 'more', "needn't", "you're", 'yours', 'them', 'now', '=', '^', 'yourselves', 'themselves', 'between', 'his', '}', "hadn't", 'mustn', 'up', 'each', 'had', "that'll", 'm', 'are', 'our', 's', '"', 'won', 'about', 'through', 'some', 'then', 'do', 'here', 'such', '-', 'its', 'who', '_', 'him', 'other', 'to', 'off', "wouldn't", "doesn't", 'itself', 'above', 'very', 'doesn', 'all', 'did', '>', 'wouldn', 'on', 'a', "weren't", '~', 'own', 'there', 'not', 'can', 'few', '&', 'if', 'shan', 'me', 'it', 'at', 'why', 'ours', 'having', "isn't", 're', "aren't", 'they', 'where', 'until', 'don', 'or', 'were', 'you', 'against', 'below', "should've", 'didn', 'been', 'most', '.'}
lemmatizer = WordNetLemmatizer()

model_input=""
model_output=""

def pos_for_lemmatizer(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_reviews(review_words):
    clean_review_words = []
    for word in review_words:
        if word.lower() not in stop_words:
            pos = pos_for_lemmatizer(pos_tag([word])[0][1])
            clean_word = lemmatizer.lemmatize(word, pos=pos)
            clean_review_words.append(clean_word.lower())
    return clean_review_words

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('base.html')



@app.route('/predict', methods=['POST'])
def predict():
    global model_input
    global model_output
    if request.method == 'POST':
        text = request.form['message']
        tokens = nltk.word_tokenize(text)
 
        # print(tokens)
        out  = " ".join(clean_reviews(tokens))
        # print(out)
        vect = cv.transform([out])
        my_prediction = clf.predict(vect)[0]
        model_input = text
        model_output = my_prediction
        # print(model_output)
    return render_template('base.html',prediction = model_output)

# route for incremental training of model
@app.route('/save_pred', methods=['POST'])
def save_pred():
    # retrieve global variables
    global model_input
    global model_output
    tokens = nltk.word_tokenize(model_input)
    out  =  " ".join(clean_reviews(tokens))
    # vectorize user input
    final_features = cv.transform([out])
    # get user's button choice (correct/incorrect)
    save_type = request.form["save_type"]

    # return text
    return_text = "The weights were strengthened, thank you for teaching me!"
    
    # modify global variable if user selected "incorrect" for retraining
    if(save_type == 'incorrect'):
        return_text = "The weights were changed, thank you for correcting me!"
        if(model_output == 'pos'):
            model_output = 'neg'
        else:
            model_output = 'pos'
        # else:
        #     print("Error: Model output was neither Neg nor Pos")
    

    # Strengthen weight of particular connection
    max_iter = 100
    counter = 0
    for i in range (0,max_iter):
        clf.partial_fit(final_features, [model_output])
        if(clf.predict(final_features) == [model_output]):
            counter = i
            break
    
    # Save trained model pickle
    joblib.dump(clf, ('gnb.pkl'))
    
    # fields inside CSV to store for retrain verification
    fields = [model_input, model_output, counter]
    
    #retrain model
    with open(('user_teaching_data.csv'), 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
    
    # return confirmation code for user
    return return_text



if __name__ == '__main__':
    app.run(debug = True)