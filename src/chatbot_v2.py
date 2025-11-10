import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import os
import json
import re
import requests
import secrets
import sqlite3
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/models/llama-v3p3-70b-instruct"

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "your-email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your-app-password")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "your-email@gmail.com")

# Admin password
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

MAX_QUESTIONS = 10  # Changed from 4 to 10
OUTPUT_DIR = "results"
DB_PATH = "interviews.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Database Setup & Functions
# --------------------------
def init_database():
    """Initializes the interviews table."""
    print("Initializing database...")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interviews
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 candidate_name TEXT NOT NULL,
                 candidate_email TEXT,
                 token TEXT UNIQUE NOT NULL,
                 status TEXT DEFAULT 'pending',
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 started_at TIMESTAMP,
                 completed_at TIMESTAMP,
                 expires_at TIMESTAMP,
                 ip_address TEXT,
                 result_filepath TEXT,
                 evaluation_json_data TEXT,
                 average_score REAL,
                 recommendation TEXT,
                 full_name TEXT,
                 age INTEGER,
                 years_experience REAL,
                 location TEXT,
                 notice_period TEXT,
                 expected_salary TEXT,
                 specialist TEXT
                )''')
    conn.commit()
    conn.close()
    print("Database initialized")  # Debug print

def add_new_columns_if_not_exists():
    """Ensures new schema columns exist in an existing database."""
    print("Adding new columns if needed...")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute("SELECT result_filepath FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN result_filepath TEXT")
        conn.commit()
    try:
        c.execute("SELECT evaluation_json_data FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN evaluation_json_data TEXT")
        conn.commit()
    try:
        c.execute("SELECT average_score FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN average_score REAL")
        conn.commit()
    try:
        c.execute("SELECT recommendation FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN recommendation TEXT")
        conn.commit()
    try:
        c.execute("SELECT full_name FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN full_name TEXT")
        conn.commit()
    try:
        c.execute("SELECT age FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN age INTEGER")
        conn.commit()
    try:
        c.execute("SELECT years_experience FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN years_experience REAL")
        conn.commit()
    try:
        c.execute("SELECT location FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN location TEXT")
        conn.commit()
    try:
        c.execute("SELECT notice_period FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN notice_period TEXT")
        conn.commit()
    try:
        c.execute("SELECT expected_salary FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN expected_salary TEXT")
        conn.commit()
    try:
        c.execute("SELECT specialist FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN specialist TEXT")
        conn.commit()
    
    conn.close()
    print("Column checks complete")  # Debug print

def reset_database():
    """Deletes all records from the interviews table."""
    print("Resetting database...")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM interviews')
    conn.commit()
    conn.close()
    print("Database reset complete")  # Debug print
    return True

def generate_interview_token(candidate_name, candidate_email="", hours_valid=48):
    """Generate unique token and store in database"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=hours_valid)
    
    print(f"Generating token for {candidate_name}")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO interviews (candidate_name, candidate_email, token, expires_at)
                 VALUES (?, ?, ?, ?)''',
              (candidate_name, candidate_email, token, expires_at))
    conn.commit()
    conn.close()
    print(f"Token generated: {token}")  # Debug print
    
    return token

def validate_token(token):
    """Validate token; expire immediately after first use."""
    print(f"Validating token: {token}")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, candidate_name, status, expires_at, completed_at 
                 FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()

    if not result:
        print("Token not found in database")  # Debug print
        conn.close()
        return {"valid": False, "message": "❌ Invalid interview link"}

    interview_id, candidate_name, status, expires_at, completed_at = result
    print(f"Found interview for {candidate_name}")  # Debug print

    # Check if already completed or expired
    expires_dt = datetime.fromisoformat(expires_at)
    if status == 'completed' or completed_at:
        print("Interview already completed")  # Debug print
        conn.close()
        return {"valid": False, "message": "❌ This interview has already been completed"}
    if datetime.now() > expires_dt:
        print("Interview link expired")  # Debug print
        conn.close()
        return {"valid": False, "message": "❌ This interview link has expired"}

    # ✅ Don't mark as completed during validation - mark during completion
    conn.close()
    print("Token validation successful")  # Debug print

    return {"valid": True, "interview_id": interview_id, "candidate_name": candidate_name}
    
def get_candidate_profile(token):
    """Get candidate profile from database"""
    print(f"Getting profile for token: {token}")  # Debug print
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT full_name, age, years_experience, location, notice_period, expected_salary, specialist
                 FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()
    conn.close()

    if not result or not any(result):  # If no profile or all fields are None
        print("No profile found")  # Debug print
        return None

    full_name, age, years_exp, location, notice, exp_salary, specialist = result
    if not all([full_name, age, years_exp, location, notice, exp_salary, specialist]):
        print("Incomplete profile found")  # Debug print
        return None  # Return None if any required field is missing

    profile = {
        "full_name": full_name,
        "age": age,
        "years_experience": years_exp,
        "location": location,
        "notice_period": notice,
        "expected_salary": exp_salary,
        "specialist": specialist
    }
    print(f"Found profile: {profile}")  # Debug print
    return profile

[... Rest of the code remains the same, but without the app definition and with debug prints added ...]

# --------------------------
# Layouts
# --------------------------
def create_header():
    return dbc.Container([
        html.Div([
            html.H1("HR Interview Evaluator", className="display-5 fw-bold mb-0", style={"color": "#B68648"}),
            html.P("Alandalusia Health Egypt",
                   className="lead mb-0 opacity-75",
                   style={"color": "#B68648"})
        ], style={
            "backgroundColor": "#F2F2F2",
            "padding": "20px",
            "textAlign": "center",
            "borderRadius": "0 0 20px 20px"
        })
    ], fluid=True, className="p-0 mb-4")

def interview_layout():
    """Candidate interview interface"""
    print("Creating interview layout")  # Debug print
    return dbc.Container([
        create_header(),
        dcc.Store(id='questions-store', data=[]),
        dcc.Store(id='answers-store', data={}),
        dcc.Store(id='current-index-store', data=0),
        dcc.Store(id='step-store', data=0),
        dcc.Store(id='token-store', data=''),
        dcc.Store(id='candidate-store', data=''),
        dcc.Store(id='evaluation-store', data=None),
        dcc.Store(id='profile-store', data={}),
        html.Div(id="token-validation-result"),
        html.Div(id="submit-feedback", className="mt-2"),
        html.Div(id="interview-section"),
        html.Div(id="evaluation-section")
    ], fluid=True, className="p-4")

def register_callbacks(app):
    """Register all callbacks with the provided Dash app"""
    print("Registering interview callbacks")  # Debug print
    
    @app.callback(
        [Output('token-store', 'data'),
         Output('candidate-store', 'data'),
         Output('token-validation-result', 'children'),
         Output('questions-store', 'data'),
         Output('step-store', 'data')],
        [Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def validate_interview_token(pathname):
        print(f"Validating token from pathname: {pathname}")
        # ... [Rest of the validate_interview_token function] ...
    
    @app.callback(
        [Output('profile-store', 'data'),
         Output('questions-store', 'data', allow_duplicate=True),
         Output('step-store', 'data', allow_duplicate=True),
         Output('submit-feedback', 'children', allow_duplicate=True)],
        [Input('submit-profile-btn', 'n_clicks')],
        [State('full-name-input', 'value'),
         State('age-input', 'value'),
         State('years-experience-input', 'value'),
         State('location-input', 'value'),
         State('notice-period-input', 'value'),
         State('expected-salary-input', 'value'),
         State('specialist-input', 'value'),
         State('token-store', 'data')],
        prevent_initial_call=True
    )
    def submit_profile(n_clicks, full_name, age, years_experience, location, notice_period, expected_salary, specialist, token):
        # ... [Rest of the submit_profile function] ...
        pass
    
    # ... Register other callbacks similarly
    
    print("Interview callbacks registered")  # Debug print
    return app