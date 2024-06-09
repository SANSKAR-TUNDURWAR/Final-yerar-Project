# from flask import Flask, request, render_template, redirect, url_for, jsonify, session, send_file
# from PIL import Image
# import numpy as np
# import pickle
# import json
# import sqlite3
# import cv2
# import matplotlib.pyplot as plt
# from io import BytesIO
# import hashlib
# import io

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Set a secret key for the session


# def predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
#     r_d_expenses = float(r_d_expenses)
#     administration_expenses = float(administration_expenses)
#     marketing_expenses = float(marketing_expenses)
    
#     if r_d_expenses < 0 or administration_expenses < 0 or marketing_expenses < 0:
#         return "Invalid input: expenses cannot be negative"

#     if r_d_expenses == 0 and administration_expenses == 0 and marketing_expenses == 0:
#         return 0

#     with open('models/startp_profit_prediction_lr_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     with open("models/columns.json", "r") as f:
#         data_columns = json.load(f)['data_columns']

#     try:
#         state_index = data_columns.index('state_' + str(state).lower())
#     except:
#         state_index = -1

#     x = np.zeros(len(data_columns))
#     x[0] = r_d_expenses
#     x[1] = administration_expenses
#     x[2] = marketing_expenses
#     if state_index >= 0:
#         x[state_index] = 1

#     predicted_profit = round(model.predict([x])[0], 2)

#     return {
#         'r_d_expenses': r_d_expenses,
#         'administration_expenses': administration_expenses,
#         'marketing_expenses': marketing_expenses,
#         'predicted_profit': predicted_profit
#     }

# def generate_bar_graph(data):
#     fig, ax = plt.subplots()
#     categories = list(data.keys())[:-1]
#     values = list(data.values())[:-1]

#     ax.bar(categories, values)
#     ax.set_title('Startup Expenses')
#     ax.set_ylabel('Amount in $')
#     ax.set_xlabel('Expense Category')

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     return buf


# # Database initialization
# def init_db():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS users
#              (id INTEGER PRIMARY KEY AUTOINCREMENT,
#               username TEXT UNIQUE,
#               password_hash TEXT,
#               email VARCHAR,
#               phone_no INTEGER,
#               R_address VARCHAR(255),
#               gender VARCHAR,
#               age INTEGER,
#               dob DATE)''')

#     conn.commit()
#     conn.close()

# init_db()

# @app.route('/', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         try:
#             username = request.form['username']
#             password = request.form['U_password']
            
#             # Hash the password for comparison
#             hashed_password = hashlib.sha256(password.encode()).hexdigest()

#             conn = sqlite3.connect('users.db')
#             c = conn.cursor()
#             c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
#             user = c.fetchone()
#             conn.close()
            
#             if user:
#                 # Store user details in session
#                 session['user'] = {
#                     'id': user[0],
#                     'username': user[1],
#                     'email': user[3],
#                     'phone_no': user[4],
#                     'R_address': user[5],
#                     'gender': user[6],
#                     'age': user[7],
#                     'dob': user[8]
#                 }
#                 # Redirect to home page
#                 return redirect('/info')
#             else:
#                 # Invalid credentials, render login page with error message
#                 return render_template('login1.html', error='Invalid username or password')
        
#         except Exception as e:
#             # Handle any exceptions
#             return render_template('error.html', message="An error occurred during login. Please try again later.")

#     # If it's a GET request, render the login page
#     return render_template('login1.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         try:
#             username = request.form['username']
#             password = request.form['U_password']
#             email = request.form['email']
#             phone = request.form['phone_no']
#             R_address = request.form['R_address']
#             gender = request.form['gender']
#             age = request.form['age']
#             dob = request.form['dob']

#             # Hash the password
#             password_hash = hashlib.sha256(password.encode()).hexdigest()

#             conn = sqlite3.connect('users.db')
#             c = conn.cursor()
#             c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
#             conn.commit()
#             conn.close()

#             return "Registration successful!", 200

#         except Exception as e:
#             print("Error during registration:", e)
#             return "An error occurred during registration. Please try again later.", 500

#     return render_template('register1.html')

# @app.route("/info")
# def info():
#     # Retrieve user details from session
#     user = session.get('user', None)
#     if user:
#         return render_template("info.html", user=user)
#     else:
#         # Redirect to login page if user is not logged in
#         return redirect(url_for('login'))

# @app.route("/government_schemes")
# def government_schemes():
#     return render_template('government_schemes.html')    



# @app.route("/home")
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         r_d_expenses = request.form['r_d_expenses']
#         administration_expenses = request.form['administration_expenses']
#         marketing_expenses = request.form['marketing_expenses']
#         state = request.form['state']
#         output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
        
#         if isinstance(output, str):
#             return render_template('result.html', show_hidden=True, prediction_text=output)
        
#         graph = generate_bar_graph(output)
#         return render_template('result.html', show_hidden=True, prediction_text='Startup Business Profit must be ${}'.format(output['predicted_profit']), graph_url="/plot.png")

# @app.route('/plot.png')
# def plot_png():
#     r_d_expenses = request.args.get('r_d_expenses')
#     administration_expenses = request.args.get('administration_expenses')
#     marketing_expenses = request.args.get('marketing_expenses')
#     state = request.args.get('state')
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
#     graph = generate_bar_graph(output)
#     return send_file(graph, mimetype='image/png')



# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, redirect, url_for, jsonify, session, send_file, send_from_directory, make_response
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
import json
import sqlite3
# import cv2
import matplotlib.pyplot as plt
import io
import os
import hashlib
import io
import csv
import base64
import requests
import certifi
import re
from datetime import datetime
import datetime
import pandas as pd
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# def predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
#     r_d_expenses = float(r_d_expenses)
#     administration_expenses = float(administration_expenses)
#     marketing_expenses = float(marketing_expenses)
    
#     if r_d_expenses < 0 or administration_expenses < 0 or marketing_expenses < 0:
#         return "Invalid input: expenses cannot be negative"

#     if r_d_expenses == 0 and administration_expenses == 0 and marketing_expenses == 0:
#         return 0

#     with open('models/startup_profit_prediction_lr_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     with open("models/columns.json", "r") as f:
#         data_columns = json.load(f)['data_columns']

#     # Find the index corresponding to the state feature
#     try:
#         state_index = data_columns.index('state_' + str(state).lower())
#     except ValueError:
#         state_index = -1

#     x = np.zeros(len(data_columns))
#     x[0] = r_d_expenses
#     x[1] = administration_expenses
#     x[2] = marketing_expenses
#     if state_index >= 0:
#         x[state_index] = 1

#     predicted_profit = round(model.predict([x])[0], 2)

#     return {
#         'r_d_expenses': r_d_expenses,
#         'administration_expenses': administration_expenses,
#         'marketing_expenses': marketing_expenses,
#         'predicted_profit': predicted_profit
#     }


def predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
    r_d_expenses = float(r_d_expenses)
    administration_expenses = float(administration_expenses)
    marketing_expenses = float(marketing_expenses)
    
    if r_d_expenses < 0 or administration_expenses < 0 or marketing_expenses < 0:
        return "Invalid input: expenses cannot be negative"

    if r_d_expenses == 0 and administration_expenses == 0 and marketing_expenses == 0:
        return {
            'r_d_expenses': r_d_expenses,
            'administration_expenses': administration_expenses,
            'marketing_expenses': marketing_expenses,
            'predicted_profit': 0
        }

    with open('models/startup_profit_prediction_lr_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("models/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    # Find the index corresponding to the state feature
    try:
        state_index = data_columns.index('state_' + str(state).lower())
    except ValueError:
        state_index = -1

    x = np.zeros(len(data_columns))
    x[0] = r_d_expenses
    x[1] = administration_expenses
    x[2] = marketing_expenses
    if state_index >= 0:
        x[state_index] = 1

    predicted_profit = round(model.predict([x])[0], 2)

    return {
        'r_d_expenses': r_d_expenses,
        'administration_expenses': administration_expenses,
        'marketing_expenses': marketing_expenses,
        'predicted_profit': predicted_profit
    }


# def predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
#     r_d_expenses = float(r_d_expenses)
#     administration_expenses = float(administration_expenses)
#     marketing_expenses = float(marketing_expenses)
    
#     if r_d_expenses < 0 or administration_expenses < 0 or marketing_expenses < 0:
#         return "Invalid input: expenses cannot be negative"

#     if r_d_expenses == 0 and administration_expenses == 0 and marketing_expenses == 0:
#         return {
#             'r_d_expenses': r_d_expenses,
#             'administration_expenses': administration_expenses,
#             'marketing_expenses': marketing_expenses,
#             'predicted_profit': 0,
#             'feature_importances': [0, 0, 0, 0]  # Assuming 4 features, update as necessary
#         }

#     with open('models/startup_profit_prediction_lr_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     with open("models/columns.json", "r") as f:
#         data_columns = json.load(f)['data_columns']

#     # Find the index corresponding to the state feature
#     try:
#         state_index = data_columns.index('state_' + str(state).lower())
#     except ValueError:
#         state_index = -1

#     x = np.zeros(len(data_columns))
#     x[0] = r_d_expenses
#     x[1] = administration_expenses
#     x[2] = marketing_expenses
#     if state_index >= 0:
#         x[state_index] = 1

#     predicted_profit = round(model.predict([x])[0], 2)
    
#     # Get feature importances if the model supports it
#     try:
#         feature_importances = model.feature_importances_.tolist()
#     except AttributeError:
#         feature_importances = [0, 0, 0, 0]  # Dummy values, update as necessary

#     return {
#         'r_d_expenses': r_d_expenses,
#         'administration_expenses': administration_expenses,
#         'marketing_expenses': marketing_expenses,
#         'predicted_profit': predicted_profit,
#         'feature_importances': feature_importances
#     }


# def generate_bar_graph(data):
#     fig, ax = plt.subplots()
#     categories = list(data.keys())[:-1]
#     values = list(data.values())[:-1]

#     ax.bar(categories, values)
#     ax.set_title('Startup Expenses')
#     ax.set_ylabel('Amount in $')
#     ax.set_xlabel('Expense Category')

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     return buf

def generate_bar_graph(data):
    fig, ax = plt.subplots()
    categories = list(data.keys())[:-1]
    values = list(data.values())[:-1]

    ax.bar(categories, values)
    ax.set_title('Startup Expenses')
    ax.set_ylabel('Amount in $')
    ax.set_xlabel('Expense Category')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free up memory
    return buf


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password_hash TEXT,
              email VARCHAR,
              phone_no INTEGER,
              R_address VARCHAR(255),
              gender VARCHAR,
              age INTEGER,
              dob DATE)''')

    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  time TEXT,
                  date INTEGER,
                  month INTEGER,
                  year INTEGER,
                  prediction_result TEXT,
                  r_d_expense TEXT,
                  administration_expenses TEXT,
                  marketing_expenses TEXT,
                  state TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            
            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Store user details in session
                session['user'] = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'phone_no': user[4],
                    'R_address': user[5],
                    'gender': user[6],
                    'age': user[7],
                    'dob': user[8]
                }
                # Redirect to home page
                return redirect('/info')
            else:
                # Invalid credentials, render login page with error message
                return render_template('login1.html', error='Invalid username or password')
        
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="An error occurred during login. Please try again later.")

    # If it's a GET request, render the login page
    return render_template('login1.html')

#original
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            email = request.form['email']
            phone = request.form['phone_no']
            R_address = request.form['R_address']
            gender = request.form['gender']
            age = request.form['age']
            dob = request.form['dob']

            # Hash the password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
            conn.commit()
            conn.close()

            return "Registration successful!", 200

        except Exception as e:
            print("Error during registration:", e)
            return "An error occurred during registration. Please try again later.", 500

    return render_template('register1.html')


@app.route("/info")
def info():
    user = session.get('user', None)
    if user:
        return render_template("info.html", user=user)
    else:
        return redirect(url_for('login'))

@app.route("/government_schemes")
def government_schemes():
    return render_template('government_schemes.html')    

@app.route("/home")
def home():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     r_d_expenses = request.form['r_d_expenses']
#     administration_expenses = request.form['administration_expenses']
#     marketing_expenses = request.form['marketing_expenses']
#     state = request.form['state']
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
#     if isinstance(output, str):
#         return render_template('result.html', show_hidden=True, prediction_text=output)
    
#     return render_template('result.html', show_hidden=True, prediction_text=f'Startup Business Profit must be ${output["predicted_profit"]}', r_d_expenses=r_d_expenses, administration_expenses=administration_expenses, marketing_expenses=marketing_expenses, state=state)

# @app.route('/predict', methods=['POST'])
# def predict():
#     r_d_expenses = request.form['r_d_expenses']
#     administration_expenses = request.form['administration_expenses']
#     marketing_expenses = request.form['marketing_expenses']
#     state = request.form['state']
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
#     if isinstance(output, str):
#         return render_template('result.html', show_hidden=True, prediction_text=output)
    
#     return render_template('result.html', show_hidden=True, prediction_text=f'Startup Business Profit must be ${output["predicted_profit"]}', r_d_expenses=r_d_expenses, administration_expenses=administration_expenses, marketing_expenses=marketing_expenses, state=state)


# @app.route('/predict', methods=['POST'])
# def predict():
#     r_d_expenses = request.form['r_d_expenses']
#     administration_expenses = request.form['administration_expenses']
#     marketing_expenses = request.form['marketing_expenses']
#     state = request.form['state']
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
#     # Prepare data for CSV
#     csv_data = io.StringIO()
#     csv_writer = csv.writer(csv_data)
#     csv_writer.writerow(['R&D Expenses', 'Administration Expenses', 'Marketing Expenses', 'State', 'Predicted Profit'])
#     if isinstance(output, str):
#         csv_writer.writerow([r_d_expenses, administration_expenses, marketing_expenses, state, output])
#     else:
#         csv_writer.writerow([r_d_expenses, administration_expenses, marketing_expenses, state, output["predicted_profit"]])
    
#     csv_data.seek(0)
    
#     # Provide CSV download link
#     return render_template('result1.html', show_hidden=True, prediction_text=f'Startup Business Profit must be ${output["predicted_profit"]}' if not isinstance(output, str) else output,
#                            r_d_expenses=r_d_expenses, administration_expenses=administration_expenses,
#                            marketing_expenses=marketing_expenses, state=state,
#                            csv_data=csv_data.getvalue())



# @app.route('/download_csv', methods=['GET'])
# def download_csv():
#     csv_data = request.args.get('csv_data')
#     return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv', as_attachment=True, download_name='prediction_result.csv')


# @app.route('/predict', methods=['POST'])
# def predict():
#     r_d_expenses = float(request.form['r_d_expenses'])
#     administration_expenses = float(request.form['administration_expenses'])
#     marketing_expenses = float(request.form['marketing_expenses'])
#     state = request.form['state']
    
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
#     plot_feature_importance(output["feature_importances"])
    
#     # Example true and predicted values, replace with actual values from your data
#     y_true = np.array([100, 150, 200])  # Replace with actual profits
#     y_pred = np.array([95, 155, 210])   # Replace with model's predicted profits
#     plot_actual_vs_predicted(y_true, y_pred)
    
#     if isinstance(output, str):
#         return render_template('result.html', show_hidden=True, prediction_text=output)
    
#     return render_template('result.html', show_hidden=True, prediction_text=f'Startup Business Profit must be ${output["predicted_profit"]}',
#                            r_d_expenses=r_d_expenses, administration_expenses=administration_expenses,
#                            marketing_expenses=marketing_expenses, state=state, feature_importance_plot='static/feature_importance.png',
#                            actual_vs_predicted_plot='static/actual_vs_predicted.png')

def store_prediction(user_id, predicted_profit, r_d_expenses, administration_expenses, marketing_expenses, state):
    time = datetime.datetime.now().strftime("%H:%M:%S")
    date = datetime.datetime.now().strftime('%d-%b-%Y')
    month = datetime.datetime.now().month
    year = datetime.datetime.now().year
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''INSERT INTO predictions (user_id, time, date, month, year, prediction_result, r_d_expense, administration_expenses, marketing_expenses, state)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, time, date, month, year, predicted_profit, r_d_expenses, administration_expenses, marketing_expenses, state))
    
    conn.commit()
    conn.close()



@app.route('/predict', methods=['POST'])
def predict():
    r_d_expenses = request.form['r_d_expenses']
    administration_expenses = request.form['administration_expenses']
    marketing_expenses = request.form['marketing_expenses']
    state = request.form['state']
    output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
    # Prepare data for CSV
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['R&D Expenses', 'Administration Expenses', 'Marketing Expenses', 'State', 'Predicted Profit'])
    if isinstance(output, str):
        csv_writer.writerow([r_d_expenses, administration_expenses, marketing_expenses, state, output])
    else:
        csv_writer.writerow([r_d_expenses, administration_expenses, marketing_expenses, state, output["predicted_profit"]])
    
    csv_data.seek(0)


    #suggestions



    
    def generate_profit_suggestions(r_d_expenses, administration_expenses, marketing_expenses, state):        
        prompt = f"""
        Provide concise, numbered suggestions for improving profit of startup based on these attributes:
        - R&D Expenses: {r_d_expenses}
        - administration_expenses: {administration_expenses}
        - marketing_expenses: {marketing_expenses}
        - state: {state}
        Keep the suggestions short and numbered.
        """

    
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            data=json.dumps(data),
            
            verify=False
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            suggestions = response_data['content'][0]['text'].strip()
            suggestions_list = [re.sub(r'^\d+\.\s*', '', suggestion).strip() for suggestion in suggestions.split('\n') if suggestion.strip()]

        else:
            suggestions_list = "Failed to generate suggestions."
        return suggestions_list


    # actual_vs_predicted_plot = plot_actual_vs_predicted(output["predicted_profit"],[r_d_expenses, administration_expenses, marketing_expenses, state])

    suggestion = generate_profit_suggestions(r_d_expenses, administration_expenses, marketing_expenses, state)

    #store result of prediction 
    user = session.get('user', None)
    
    store_prediction(user['id'], output["predicted_profit"], r_d_expenses, administration_expenses, marketing_expenses, state)
    # get_last_six_months_data()
    # print(getUserPredictionHistory(user['id']))

    
    # Provide CSV download link
    return render_template('result1.html', show_hidden=True, prediction_text=f'Startup Business Profit must be ${output["predicted_profit"]}' if not isinstance(output, str) else output,
                           r_d_expenses=r_d_expenses, administration_expenses=administration_expenses,
                           marketing_expenses=marketing_expenses, state=state,
                        #    actual_vs_predicted_plot=actual_vs_predicted_plot,
                           suggestions =suggestion,
                           csv_data=csv_data.getvalue())



@app.route('/download_csv', methods=['GET'])
def download_csv():
    csv_data = request.args.get('csv_data')
    return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv', as_attachment=True, download_name='prediction_result.csv')


@app.route('/insight')
def insight():
    # get_last_six_months_data()
    return render_template('insight.html')



@app.route('/last-six-months-data', methods=['POST'])
def get_last_six_months_data():
    files = []
    user = session.get('user', None)

    # Get user ID from session
    user_id = user['id']
    if not user_id:
        return jsonify({'error': 'User not logged in'}), 401

    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # Connect to the database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    for i in range(6):
        target_month = current_month - i
        target_year = current_year
        if target_month <= 0:
            target_month += 12
            target_year -= 1

        c.execute('''
            SELECT * FROM predictions
            WHERE month = ? AND year = ? AND user_id = ?
        ''', (target_month, target_year, user_id))
        data_exists = c.fetchone()

        if data_exists:
            c.execute('''
                SELECT date, time, prediction_result, r_d_expense, administration_expenses, 
                       marketing_expenses, state
                FROM predictions
                WHERE month = ? AND year = ? AND user_id = ?
            ''', (target_month, target_year, user_id))
            rows = c.fetchall()

            columns = ['date', 'time', 'prediction_result', 'r_d_expense', 'administration_expenses',
                       'marketing_expenses', 'state']
            current_month_data_df = pd.DataFrame(rows, columns=columns)
            month_year_str = datetime.datetime(target_year, target_month, 1).strftime('%b%Y')
            csv_file_name = f'predictions_data_{month_year_str}.csv'
            # File path
            csv_file_path = os.path.join(app.root_path, csv_file_name)
            current_month_data_df.to_csv(csv_file_path, index=False)

            files.append({'file_name': csv_file_name, 'file_path': csv_file_path})

    conn.close()
    # return render_template('insight.html', csv_file = csv_file_name.getvalue())
    return jsonify(files)



# @app.route('/download-prediction-csv', methods=['GET'])
# def download_prediction_csv():
#     month_year_str = datetime.datetime(target_year, target_month, 1).strftime('%b%Y')
#     csv_file_name = f'predictions_data_{month_year_str}.csv'
#     filename = os.path.basename(filename)
#     response = make_response(send_file(filename, as_attachment=True))
#     response.headers['Cache-Control'] = 'no-cache'
#     return response


@app.route('/download-prediction', methods=['GET'])
def download_file():
    target_month = datetime.datetime.now().month
    target_year = datetime.datetime.now().year
    month_year_str = datetime.datetime(target_year, target_month, 1).strftime('%b%Y')
    csv_file_name = f'predictions_data_{month_year_str}.csv'
    file_path = os.path.join(app.root_path, csv_file_name)

    print(csv_file_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404




GRAPH_FOLDER = 'graphs'
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER

@app.route('/generate-graphs', methods=['POST'])
def generate_graphs():
    attribute = request.json.get('attribute')


    # Check if attributes are provided
    if not attribute:
        return jsonify({'error': 'Attributes not provided'}), 400

    graph_dir = os.path.join(app.root_path, app.config['GRAPH_FOLDER'])
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # List all files in the static directory
    existing_files = os.listdir(graph_dir)

    # Iterate over existing files and remove graphs
    for file in existing_files:
        if file.endswith('.png'):  # Check if file is a PNG image
            os.remove(os.path.join(graph_dir, file))  # Delete the file

     # Save the graphs in the graph folder
    line_plot_path = os.path.join(graph_dir, 'line_plot.png')
    histogram_path = os.path.join(graph_dir, 'histogram.png')
    box_plot_path = os.path.join(graph_dir, 'box_plot.png')
    heatmap_path = os.path.join(graph_dir, 'heatmap.png')
    count_plot_path = os.path.join(graph_dir,'count_plot.png')
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # Generate the file name using the current year and month
    month_year_str = datetime.datetime(current_year, current_month, 1).strftime('%b%Y')
    # print("in backedn month year",month_year_str)
    csv_file_name = f'predictions_data_{month_year_str}.csv'
    csv_file_path = os.path.join(app.root_path, csv_file_name)

    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
    else:
        return jsonify({'error': 'CSV file not found'}), 404
  

    
    numeric_columns = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_columns]
    df = df.sort_values(by='date', ascending=True)
    # print(df)


    plt.figure(figsize=(13, 10))
    plt.plot(df['date'], df[attribute], label=attribute)
    plt.xlabel('Date')
    plt.ylabel('Attribute Value')
    plt.title('Time Series Analysis')
    plt.legend()
    plt.savefig(line_plot_path)
    plt.close()

    # Histogram for attribute distribution
    plt.figure(figsize=(13, 10))
    sns.histplot(df[attribute], kde=True, color='blue', label=attribute)
    plt.xlabel('Attribute Value')
    plt.ylabel('Frequency')
    plt.title('Attribute Distribution')
    plt.legend()
    plt.savefig(histogram_path)
    plt.close()

    # Box plot for attribute variation
    plt.figure(figsize=(13, 10))
    sns.boxplot(data=df[[attribute]])
    plt.ylabel('Attribute Value')
    plt.title('Attribute Variation')
    plt.savefig(box_plot_path)
    plt.close()

    plt.figure(figsize=(13,10))
    plt.title("Attribute of expenses",fontsize=15)
    c1=sns.countplot(x=attribute,data=df,palette="deep")
    c1.bar_label(c1.containers[0],size=12)
    plt.xticks(rotation=45)
    plt.savefig(count_plot_path)

    # Heatmap for correlation matrix
    plt.figure(figsize=(13, 10))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(heatmap_path)
    plt.close()

    return jsonify({
        'linePlotUrl': 'line_plot.png',
        'histogramUrl': 'histogram.png',
        'boxPlotUrl':  'box_plot.png',
        'countPlotUrl': 'count_plot.png',
        'heatmapUrl': 'heatmap.png'
    })

@app.route('/graphs/<path:filename>')
def send_graphs(filename):
    return send_from_directory(app.config['GRAPH_FOLDER'], filename)


# Define the PDF folder path
PDF_FOLDER = "pdf_files"
app.config['PDF_FOLDER'] = PDF_FOLDER
# app.config['PDF_FOLDER'] = os.path.join(app.root_path, 'pdf_files')
@app.route('/generate-graphs-pdf', methods=['GET'])
def generate_graphs_pdf():
    attribute = request.args.get('attribute')
    print(attribute)
    # Create a PDF file
    pdf_filename = f"{attribute}_graphs.pdf"
    pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_filename)
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Get all image files from the graphs folder
    graph_folder = app.config['GRAPH_FOLDER']
    graph_files = [f for f in os.listdir(graph_folder) if os.path.isfile(os.path.join(graph_folder, f))]

    # Draw each image on a separate page in the PDF
    for i, graph_file in enumerate(graph_files):
        # Draw the image on the PDF canvas
        c.drawImage(os.path.join(graph_folder, graph_file), 50, 50, width=500, height=400)
        # Add a new page for the next image (except for the last image)
        if i < len(graph_files) - 1:
            c.showPage()

    # Save the PDF
    c.save()

    # Send the PDF file to the user for download
    return send_file(pdf_path, as_attachment=True)
















def getUserPredictionHistory(userid):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM predictions')
    rows = c.fetchall()
    conn.close()
    return rows



# @app.route('/plot.png')
# def plot_png():
#     r_d_expenses = request.args.get('r_d_expenses')
#     administration_expenses = request.args.get('administration_expenses')
#     marketing_expenses = request.args.get('marketing_expenses')
#     state = request.args.get('state')
#     output = predict_startup_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
    
#     graph = generate_bar_graph(output)
#     return send_file(graph, mimetype='image/png')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = session.get('user', None)
    if not user:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone_no']
        R_address = request.form['R_address']
        gender = request.form['gender']
        age = request.form['age']
        dob = request.form['dob']
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''UPDATE users SET username=?, email=?, phone_no=?, R_address=?, gender=?, age=?, dob=? WHERE id=?''',
                      (username, email, phone, R_address, gender, age, dob, user['id']))
            conn.commit()
            conn.close()
            
            # Update session
            session['user'] = {
                'id': user['id'],
                'username': username,
                'email': email,
                'phone_no': phone,
                'R_address': R_address,
                'gender': gender,
                'age': age,
                'dob': dob
            }
            return redirect(url_for('profile'))
        except sqlite3.IntegrityError:
            return render_template('profile.html', user=user, error='Username already exists.')
        except Exception as e:
            return render_template('profile.html', user=user, error=f'An error occurred: {e}')
    
    return render_template('profile.html', user=user)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
