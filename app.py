from flask import Flask, request, render_template_string, redirect, url_for, session
import joblib
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key' 
model = joblib.load('spam_model.joblib')

example_spam_keywords = ["free", "win", "winner", "claim", "urgent", "prize", "congratulations", "offer", "click", "now"]

def analyze_message(message):
    reasons = []
    text = message.strip()
    lower_text = text.lower()

    found_keywords = [word for word in example_spam_keywords if word in lower_text]
    if found_keywords:
        reasons.append("The message may contain some spam keywords.")

    total_letters = sum(1 for c in text if c.isalpha())
    uppercase_letters = sum(1 for c in text if c.isupper())
    if total_letters > 0:
        ratio = uppercase_letters / total_letters
        if ratio > 0.5:
            reasons.append("It may be because of the high uppercase letter rate.")

    exclamations = text.count('!')
    if exclamations >= 3:
        reasons.append("It may be because it contains too many exclamation marks.")

    if re.search(r'http[s]?://', text):
        reasons.append("It may be because the message contains a URL.")

    if len(text) < 10:
        reasons.append("This may be due to the message being too short.")
    elif len(text) > 300:
        reasons.append("This may be due to the message being quite long.")

    return reasons

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    reasons = []

    if request.method == 'POST':
        message = request.form.get('message', '')
        if message:
            proba = model.predict_proba([message])[0]
            prediction = model.predict([message])[0]
            confidence = max(proba)
            result = "This is a spam ❌" if prediction == 1 else "This is not spam ✅"
            reasons = analyze_message(message) if prediction == 1 else []

            session['result'] = result
            session['confidence'] = confidence
            session['reasons'] = reasons
            return redirect(url_for('home'))

    result = session.pop('result', None)
    confidence = session.pop('confidence', None)
    reasons = session.pop('reasons', [])

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Detector</title>
        <style>
            body {
                background-color: #f0f2f5;
                font-family: Arial, sans-serif;
                padding: 20px;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                min-height: 100vh;
            }
            .container {
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                max-width: 600px;
                width: 100%;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            h2 {
                text-align: center;
                margin-bottom: 20px;
            }
            textarea {
                width: 100%;
                height: 150px;
                font-size: 16px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                resize: vertical;
            }
            button {
                width: 100%;
                margin-top: 10px;
                padding: 12px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #fafafa;
            }
            .reasons {
                margin-top: 10px;
                font-size: 14px;
                color: #444;
                text-align: left;
            }
            ul {
                padding-left: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Spam Detection</h2>
            <form method="POST">
                <textarea name="message" placeholder="Type your message here..."></textarea><br><br>
                <button type="submit">Submit</button>
            </form>
            {% if result %}
            <div class="result">
                <strong>Result:</strong> {{ result }}<br>
                <strong>Confidence:</strong> %{{ "%.2f"|format(confidence * 100) }}
                {% if reasons %}
                    <div class="reasons">
                        <strong>Possible reasons:</strong>
                        <ul>
                            {% for reason in reasons %}
                                <li>{{ reason }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                {% endif %}
            </div>
            {% endif %}
                                  
            <div class="info">
            <br><br>  This tool uses a machine learning model trained on an English dataset. So please enter English input.<br>
              Results are probabilistic and are not guaranteed to be 100% accurate. After submitting, the model's confidence percentage in the result will be displayed. If it is spam, possible reasons are indicated.
              <br><br><br>
                      Bu araç, İngilizce veri kümesi üzerinde eğitilmiş bir makine öğrenimi modeli kullanır. Bu nedenle lütfen İngilizce girdi girin.<br>
              Sonuçlar olasılıksaldır ve %100 doğru oldukları garanti edilmez. Gönderdikten sonra, modelin sonuçtaki güven yüzdesi görüntülenecektir. Eğer spamsa olası nedenler belirtilir.
            </div>
        </div>
        
    </body>
    </html>
    """, result=result, confidence=confidence, reasons=reasons)

if __name__ == '__main__':
    app.run(debug=False)