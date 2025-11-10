import dash
from dash import dcc, html, Input, Output, State, callback_context, callback, no_update
import sqlite3, os, json
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import re
import requests
import secrets
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from dash import ALL
app = dash.Dash(__name__, suppress_callback_exceptions=True)
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

def add_new_columns_if_not_exists():
    """Ensures new schema columns exist in an existing database."""
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

# Run initialization and schema update
init_database()
add_new_columns_if_not_exists()

def reset_database():
    """Deletes all records from the interviews table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM interviews')
    conn.commit()
    conn.close()
    return True

def generate_interview_token(candidate_name, candidate_email="", hours_valid=48):
    """Generate unique token and store in database"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=hours_valid)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO interviews (candidate_name, candidate_email, token, expires_at)
                 VALUES (?, ?, ?, ?)''',
              (candidate_name, candidate_email, token, expires_at))
    conn.commit()
    conn.close()
    
    return token

def validate_token(token):
    """Validate token; expire immediately after first use."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, candidate_name, status, expires_at, completed_at 
                 FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()

    if not result:
        conn.close()
        return {"valid": False, "message": "❌ Invalid interview link"}

    interview_id, candidate_name, status, expires_at, completed_at = result

    # Check if already completed or expired
    expires_dt = datetime.fromisoformat(expires_at)
    if status == 'completed' or completed_at:
        conn.close()
        return {"valid": False, "message": "❌ This interview has already been completed"}
    if datetime.now() > expires_dt:
        conn.close()
        return {"valid": False, "message": "❌ This interview link has expired"}

    # ✅ Don't mark as completed during validation - mark during completion
    conn.close()

    return {"valid": True, "interview_id": interview_id, "candidate_name": candidate_name}
    
def get_candidate_profile(token):
    """Get candidate profile from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT full_name, age, years_experience, location, notice_period, expected_salary, specialist
                 FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()
    conn.close()

    if not result or not any(result):  # If no profile or all fields are None
        return None

    full_name, age, years_exp, location, notice, exp_salary, specialist = result
    if not all([full_name, age, years_exp, location, notice, exp_salary, specialist]):
        return None  # Return None if any required field is missing

    return {
        "full_name": full_name,
        "age": age,
        "years_experience": years_exp,
        "location": location,
        "notice_period": notice,
        "expected_salary": exp_salary,
        "specialist": specialist
    }

def update_candidate_profile(token, profile):
    """Update candidate profile details in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE interviews SET full_name = ?, age = ?, years_experience = ?, location = ?, 
                 notice_period = ?, expected_salary = ?, specialist = ?
                 WHERE token = ?''',
              (profile['full_name'], profile['age'], profile['years_experience'], profile['location'],
               profile['notice_period'], profile['expected_salary'], profile['specialist'], token))
    conn.commit()
    conn.close()

def mark_interview_started(token):
    """Mark interview as started"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE interviews SET started_at = ?, status = 'in_progress'
                 WHERE token = ?''', (datetime.now(), token))
    conn.commit()
    conn.close()

def mark_interview_completed(token, filepath, evaluation_json_dump, evaluation_dict):
    """Mark interview as completed and save result metadata."""
    avg_score = None
    recommendation = None
    if "final_evaluation" in evaluation_dict:
        score_str = evaluation_dict["final_evaluation"].get("average_score")
        try:
            avg_score = float(score_str)
        except (ValueError, TypeError):
            avg_score = None
        recommendation = evaluation_dict["final_evaluation"].get("recommendation")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE interviews SET completed_at = ?, 
                                       status = 'completed',
                                       result_filepath = ?,
                                       evaluation_json_data = ?,
                                       average_score = ?,
                                       recommendation = ?
                 WHERE token = ?''', 
                 (datetime.now(), filepath, evaluation_json_dump, avg_score, recommendation, token))
    conn.commit()
    conn.close()

def get_all_interviews():
    """Get all interviews for admin panel."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT candidate_name, candidate_email, token, status, 
                      created_at, completed_at, expires_at, result_filepath, evaluation_json_data,
                      average_score, recommendation, full_name, age, years_experience,
                      location, notice_period, expected_salary, specialist
                 FROM interviews ORDER BY created_at DESC''')
    results = c.fetchall()
    conn.close()
    return results

# --------------------------
# Email Notification
# --------------------------
def send_email_notification(candidate_name, filename):
    """Send email when interview is completed."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Interview Completed - {candidate_name}"
        
        body = f"""
Interview completed for: {candidate_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results file: {filename}

Check the results folder or the Admin Panel for detailed evaluation.
"""
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError:
        print(f"Email error: Failed to log in. Check EMAIL_USER and EMAIL_PASSWORD.")
        return False
    except smtplib.SMTPConnectError as e:
        print(f"Email error: Could not connect to SMTP server. Details: {e}")
        return False
    except TimeoutError as e:
        print(f"Email error: Connection timed out. Details: {e}")
        return False
    except Exception as e:
        print(f"Email error: An unexpected error occurred. Details: {e}")
        return False

# --------------------------
# Prompts & Fireworks API Functions - UPDATED FOR ALL SPECIALTIES
# --------------------------
EVALUATOR_PROMPT = """
You are an HR interviewer and evaluator.

- You will receive candidate profile and answers for 10 interview questions.
- For each answer, provide:
  Score (0–10), Strengths, Weaknesses, Suggestions for improvement.
- After all answers, provide a final overall evaluation:
  Average Score, Overall Strengths, Overall Weaknesses, and Final Recommendation (Hire / Consider / Reject).

The output must be valid JSON in the format:

{
  "evaluations": {
    "Question 1": {"score": 8, "strengths": "...", "weaknesses": "...", "suggestions": "..."},
    "Question 2": {...},
    ...
    "Question 10": {...}
  },
  "final_evaluation": {
    "average_score": 7.5,
    "overall_strengths": "...",
    "overall_weaknesses": "...",
    "recommendation": "Consider"
  }
}
Return only valid JSON, without any explanations or markdown fences.
"""

# UPDATED: Universal fallback questions for all specialties
FALLBACK_QUESTIONS = [
    "Tell me about yourself and your professional background.",
    "What are your key strengths and how do they apply to this role?",
    "What are your weak points, and how are you working to improve them?",
    "Why do you want to work for our organization?",
    "Describe a challenging situation you faced at work and how you handled it.",
    "How do you handle stress and maintain work-life balance?",
    "What are your salary expectations and notice period?",
    "Where do you see yourself professionally in the next 5 years?",
    "How do you stay updated with the latest developments in your field?",
    "Why should we hire you over other candidates?"
]

def call_fireworks_api(prompt, system_prompt="You are a helpful assistant."):
    if not FIREWORKS_API_KEY:
        print("API Key missing. Cannot call Fireworks.")
        return None

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 3000  # Increased from 2000 to handle 10 questions
    }

    try:
        response = requests.post(FIREWORKS_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Fireworks API Error: {e}")
        return json.dumps({"error": f"Fireworks API Error: {e}. Check API key and configuration."})

def clean_markdown_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()

# UPDATED: Generate questions for ANY specialty
def generate_all_questions(num=MAX_QUESTIONS, profile=None):
    """
    Generate HR interview questions tailored to candidate's specialty.
    Works for all industries: IT, Finance, Healthcare, Engineering, Marketing, etc.
    """
    prompt = f"""
    Generate {num} HR interview questions tailored to the candidate's field and specialty.
    
    Requirements:
    1. Include ONE question about the candidate's weak points/areas for improvement
    2. Focus on:
       - Professional experience and expertise in their field
       - Problem-solving and critical thinking
       - Teamwork and communication skills
       - Career goals and motivation
       - Industry-specific knowledge
       - Handling challenges and pressure
    3. Make questions relevant to their specific specialty/industry
    4. Avoid generic questions - make them insightful and meaningful
    
    Return only valid JSON like:
    {{
      "questions": [
        "Question 1 text...",
        "Question 2 text...",
        ...
        "Question {num} text..."
      ]
    }}
    """
    
    if profile:
        specialist = profile.get('specialist', 'Not specified')
        years_exp = profile.get('years_experience', 'Not specified')
        location = profile.get('location', 'Not specified')
        
        profile_context = f"""
Candidate Profile:
- Specialty/Field: {specialist}
- Years of Experience: {years_exp}
- Location: {location}
- Full Name: {profile.get('full_name', 'Not specified')}
- Age: {profile.get('age', 'Not specified')}
- Expected Salary: {profile.get('expected_salary', 'Not specified')}
- Notice Period: {profile.get('notice_period', 'Not specified')}
"""
        prompt += f"\n\nTailor the questions based on this candidate profile:\n{profile_context}"
        prompt += f"\n\nIMPORTANT: Generate questions appropriate for a {specialist} with {years_exp} years of experience."

    response = call_fireworks_api(prompt, "You are an expert HR interviewer with experience across all industries and specialties.")
    if response:
        try:
            raw = clean_markdown_fences(response)
            data = json.loads(raw)
            questions = data.get("questions", FALLBACK_QUESTIONS[:num])
            
            # Ensure we have exactly the right number of questions
            if len(questions) < num:
                questions.extend(FALLBACK_QUESTIONS[:(num - len(questions))])
            elif len(questions) > num:
                questions = questions[:num]
            
            # Ensure at least one question about weak points
            if not any("weak" in q.lower() or "weakness" in q.lower() or "improve" in q.lower() for q in questions):
                questions[2] = FALLBACK_QUESTIONS[2]  # Replace 3rd question with weak points question
            
            return questions
        except Exception as e:
            print(f"Failed to parse questions: {e}")
    
    return FALLBACK_QUESTIONS[:num]

def evaluate_with_fireworks(answers: dict, profile: dict) -> dict:
    profile_text = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in profile.items()])
    answers_text = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(answers.items())])
    full_prompt = EVALUATOR_PROMPT + "\n\nCandidate Profile:\n" + profile_text + "\n\nCandidate Answers:\n" + answers_text
    response = call_fireworks_api(full_prompt, "You are an HR interviewer for doctors.")
    
    if response:
        try:
            try:
                error_check = json.loads(response)
                if "error" in error_check:
                    return error_check
            except json.JSONDecodeError:
                pass
            raw_text = clean_markdown_fences(response)
            return json.loads(raw_text)
        except Exception as e:
            print(f"Failed to parse evaluation JSON: {e}")
            return {"error": f"Evaluation failed (JSON parsing error): {e}"}
    else:
        return {"error": "No valid response from API or response was None."}

def save_results_txt(candidate_name, answers, evaluation_json, profile):
    """Saves the comprehensive evaluation to a local TXT file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = candidate_name.replace(" ", "_") or "candidate"
    filename = f"{OUTPUT_DIR}/{safe_name}_{timestamp}.txt"
    filepath = str(Path(filename).resolve())

    lines = [f"Candidate: {candidate_name}",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "="*60, "", "Candidate Profile", "-"*60]

    for key, value in profile.items():
        lines.append(f"{key.replace('_', ' ').title()}: {value if value else '[No answer]'}")

    lines.append("")
    lines.append("Interview Answers")
    lines.append("-"*60)

    for i, (q, a) in enumerate(answers.items(), 1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a if a.strip() else '[No answer]'}")
        lines.append("")

    lines.append("Evaluation")
    lines.append("-"*60)
    
    if "error" in evaluation_json:
        lines.append(f"ERROR: Evaluation failed: {evaluation_json['error']}")
    else:
        if "evaluations" in evaluation_json:
            for q, details in evaluation_json["evaluations"].items():
                lines.append(q)
                lines.append(f"   Score: {details.get('score','N/A')}")
                lines.append(f"   Strengths: {details.get('strengths','N/A')}")
                lines.append(f"   Weaknesses: {details.get('weaknesses','N/A')}")
                lines.append(f"   Suggestions: {details.get('suggestions','N/A')}")
                lines.append("")

        if "final_evaluation" in evaluation_json:
            fe = evaluation_json["final_evaluation"]
            lines.append("Final Evaluation")
            lines.append("-"*60)
            lines.append(f"Average Score: {fe.get('average_score','N/A')}")
            lines.append(f"Overall Strengths: {fe.get('overall_strengths','N/A')}")
            lines.append(f"Overall Weaknesses: {fe.get('overall_weaknesses','N/A')}")
            lines.append(f"Recommendation: {fe.get('recommendation','N/A')}")

    content = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
        
    return filepath, content

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

def admin_panel_layout():
    """Admin panel for generating interview links and viewing results"""
    return dbc.Container([
        dbc.Card([
            dbc.CardHeader(html.H3("AI Interview Control Panel")),
            dbc.CardBody([
                html.H5("Generate New Interview Link"),
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Candidate Name *"),
                            dbc.Input(id="admin-candidate-name", type="text", placeholder="Enter name", required=True)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Candidate Email (optional)"),
                            dbc.Input(id="admin-candidate-email", type="email", placeholder="Enter email")
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Link Valid For (hours)"),
                            dbc.Input(id="admin-hours-valid", type="number", value="48", min=1, max=168)
                        ], width=4)
                    ]),
                    dbc.Button("Generate Link", id="admin-generate-link-btn", color="primary", className="mt-3"),
                    html.Div(id="admin-generated-link-output", className="mt-3"),
                ], id="admin-generate-form"),
                html.Hr(),
                html.H5("Database Management"),
                dbc.Button("⚠️ Reset Database", id="admin-reset-db-btn", color="danger", className="mt-2 mb-3"),
                html.Div(id="admin-db-reset-output", className="mb-3"),
                html.Hr(),
                html.H5("All Interviews & Results"),
                html.Div(id="admin-interviews-list"),
                dbc.Button("Refresh List", id="admin-refresh-list-btn", color="secondary", className="mt-2"),
                dcc.Download(id="admin-download-result-file")
            ])
        ])
    ], fluid=True, className="p-4")

def interview_layout():
    """Candidate interview interface"""
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

# Note: Routing and Dash app instance are handled by the main application (app.py).
# This module provides layouts and pure helper functions for the interview system
# (e.g. interview_layout, admin_panel_layout, generate_interview_token, validate_token, etc.).

# --------------------------
# Admin Panel Callbacks
# --------------------------
def handle_db_reset(n_clicks):
    if n_clicks and n_clicks > 0:
        reset_database()
        return dbc.Alert("✅ Database cleared successfully! All interview data has been removed.", color="success")
    return ""

def generate_link(n_clicks, name, email, hours, current_url):
    if not n_clicks or not name:
        return ""
    
    hours_valid = int(hours) if hours else 48
    token = generate_interview_token(name, email or "", hours_valid)
    
    base_url = current_url.split('/admin')[0] if '/admin' in current_url else current_url.rstrip('/')
    interview_url = f"{base_url}/interview/{token}"
    
    return dbc.Alert([
        html.H5("✅ Interview Link Generated!", className="mb-3"),
        html.P(f"Candidate: {name}"),
        html.P(f"Valid for: {hours_valid} hours"),
        html.Hr(),
        html.Label("Copy this link and send via WhatsApp:"),
        dbc.InputGroup([
            dbc.Input(value=interview_url, id="link-to-copy", readonly=True),
        ]),
        html.Small("Expires: " + (datetime.now() + timedelta(hours=hours_valid)).strftime('%Y-%m-%d %H:%M'), 
                   className="text-muted mt-2 d-block")
    ], color="success")

def update_interviews_list(refresh_clicks, generate_clicks, reset_output):
    interviews = get_all_interviews()
    
    if not interviews:
        return dbc.Alert("No interviews yet", color="info")
    
    table_rows = []
    for interview in interviews:
        name, email, token, status, created, completed, expires, filepath, json_data, avg_score, recommendation, full_name, age, years_exp, loc, notice, exp_sal, specialist = interview
        
        status_badge = {
            'pending': dbc.Badge("Pending", color="warning"),
            'in_progress': dbc.Badge("In Progress", color="info"),
            'completed': dbc.Badge("Completed", color="success")
        }.get(status, dbc.Badge(status, color="secondary"))
        
        score_display = f"{avg_score:.1f}" if avg_score is not None else "-"
        
        download_col = html.Td("-")
        if status == 'completed' and filepath:
            download_col = html.Td(dbc.Button(
                "Download TXT", 
                id={"type": "download-btn", "index": token},
                color="info", 
                size="sm"
            ))

        table_rows.append(html.Tr([
            html.Td(name),
            html.Td(full_name or "-"),
            html.Td(age or "-"),
            html.Td(years_exp or "-"),
            html.Td(loc or "-"),
            html.Td(exp_sal or "-"),
            html.Td(specialist or "-"),
            html.Td(email or "-"),
            html.Td(status_badge),
            html.Td(completed[:16] if completed else "-"),
            html.Td(score_display),
            html.Td(recommendation or "-"),
            download_col
        ]))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Admin Name"),
            html.Th("Full Name"),
            html.Th("Age"),
            html.Th("Experience"),
            html.Th("Location"),
            html.Th("Exp. Salary (EGP)"),
            html.Th("Specialist"),
            html.Th("Email"),
            html.Th("Status"),
            html.Th("Completed"),
            html.Th("Avg Score"),
            html.Th("Recommendation"),
            html.Th("Results")
        ])),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, striped=True, size="sm")

@app.callback(
    Output('download-result-file', 'data'),
    [Input({'type': 'download-btn', 'index': dash.ALL}, 'n_clicks')]
)
def download_admin_result(n_clicks):
    trigger = callback_context.triggered
    if not trigger or not any(n_clicks):
        return dash.no_update

    button_id = trigger[0]['prop_id'].split('.')[0]
    
    try:
        token = json.loads(button_id)['index']
    except Exception:
        print("Error parsing download button ID.")
        return dash.no_update

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT candidate_name, result_filepath FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()
    conn.close()

    if result:
        candidate_name, filepath = result
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            filename = os.path.basename(filepath)
            return dcc.send_string(content, filename=filename)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return dcc.send_string(f"Error: Could not find or read file at path: {filepath}", filename="error_download.txt")
    return dash.no_update
def validate_interview_token(pathname):
    print(f"Validating token from pathname: {pathname}")  # Debug print
    
    if not pathname or not pathname.startswith('/interview/'):
        print("No interview path")  # Debug print
        return '', '', '', [], 0
        
    try:
        # Parse token from query string if present, otherwise from path
        token = None
        if 'token=' in pathname:
            token = pathname.split('token=')[1].split('&')[0]
        else:
            token = pathname.split('/interview/')[-1]
            
        if not token:
            print("No token found")  # Debug print
            return '', '', dbc.Alert("❌ No interview token found.", color="danger"), [], 0
            
        print(f"Found token: {token}")  # Debug print
        validation = validate_token(token)
        print(f"Validation result: {validation}")  # Debug print
        
        if not validation['valid']:
            print(f"Invalid token: {validation['message']}")  # Debug print
            return '', '', dbc.Alert(validation['message'], color="danger"), [], 0
        
        welcome_msg = dbc.Alert([
            html.H4(f"Welcome, {validation['candidate_name']}! 👋"),
            html.P("Please provide your profile details to begin the interview."),
            html.P("⚠️ Note: This link can only be used once. Do not refresh the page.")
        ], color="info")
        
        profile = get_candidate_profile(token)
        print(f"Got profile from DB: {profile}")  # Debug print
        
        if profile:
            print("Profile exists, loading questions")  # Debug print
            questions = generate_all_questions(profile=profile)
            if not questions:
                print("No questions generated, using fallback")  # Debug print
                questions = FALLBACK_QUESTIONS.copy()
            step = 2  # Skip profile step
        else:
            print("No profile, using initial fallback questions")  # Debug print
            questions = FALLBACK_QUESTIONS.copy()
            step = 1  # Show profile step
        
        print(f"Returning questions: {len(questions)} and step: {step}")  # Debug print
        return token, validation['candidate_name'], welcome_msg, questions, step
        
    except Exception as e:
        print(f"Error in validate_interview_token: {str(e)}")  # Debug print
        return '', '', dbc.Alert("Error processing interview link", color="danger"), [], 0

def submit_profile(n_clicks, full_name, age, years_experience, location, notice_period, expected_salary, specialist, token):
    print(f"Submit profile called with token: {token}")  # Debug print
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not all([full_name, age, years_experience, location, notice_period, expected_salary, specialist]):
        print("Missing required fields")  # Debug print
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("⚠️ Please fill all fields before submitting.", color="warning")
    
    try:
        profile = {
            "full_name": full_name.strip(),
            "age": int(age),
            "years_experience": float(years_experience),
            "location": location.strip(),
            "notice_period": notice_period.strip(),
            "expected_salary": expected_salary.strip(),
            "specialist": specialist.strip()
        }
    except ValueError:
        print("Invalid number format")  # Debug print
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("⚠️ Please enter valid numbers for age and years of experience.", color="warning")
    
    try:
        print("Updating candidate profile in DB")  # Debug print
        update_candidate_profile(token, profile)
        print("Marking interview as started")  # Debug print
        mark_interview_started(token)
        print("Generating questions")  # Debug print
        questions = generate_all_questions(profile=profile)
        print(f"Generated {len(questions)} questions")  # Debug print
        
        if not questions:
            print("No questions generated, using fallback")  # Debug print
            questions = FALLBACK_QUESTIONS.copy()
        
        return profile, questions, 2, dbc.Alert("Profile submitted ✅ Loading questions...", color="success")
    except Exception as e:
        print(f"Error in submit_profile: {str(e)}")  # Debug print
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("❌ An error occurred. Please try again.", color="danger")

def update_interview_display(questions, current_idx, answers, step):
    interview_content, evaluation_content = [], []

    if step == 1:
        interview_content = [
            dbc.Card([
                dbc.CardHeader(html.H4("Candidate Profile")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Full Name *"),
                            dbc.Input(id="full-name-input", placeholder="Enter your full name", required=True)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Age *"),
                            dbc.Input(id="age-input", type="number", min=18, max=100, placeholder="Enter your age", required=True)
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Years of Experience *"),
                            dbc.Input(id="years-experience-input", type="number", step=0.5, placeholder="Enter years of experience", required=True)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Location *"),
                            dbc.Input(id="location-input", placeholder="Enter your location", required=True)
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Notice Period *"),
                            dbc.Input(id="notice-period-input", placeholder="Enter notice period (e.g., 30 days)", required=True)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Expected Salary (EGP) *"),
                            dbc.Input(id="expected-salary-input", placeholder="Enter expected salary (e.g., 5000 EGP)", required=True)
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Specialist *"),
                            dbc.Input(id="specialist-input", placeholder="Enter your specialty (e.g., Medical Sales)", required=True)
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Button("Submit Profile", id="submit-profile-btn", color="primary")
                ])
            ])
        ]
    elif step == 2 and questions:
        if current_idx < len(questions):
            q_text = questions[current_idx]
            interview_content = [
                dbc.Card([
                    dbc.CardHeader(html.H4(f"Question {current_idx + 1}/{MAX_QUESTIONS}")),
                    dbc.CardBody([
                        html.P(q_text, className="lead mb-3"),
                        dbc.Textarea(id="answer-input", placeholder="Type your answer here...",
                                     style={"height": "200px"}, className="mb-3",
                                     value=answers.get(q_text, "")),
                        dbc.Button("Submit Answer", id="submit-answer-btn", color="success", className="me-2"),
                    ])
                ], className="mb-3"),
                dbc.Progress(value=(len(answers)/MAX_QUESTIONS)*100, className="mb-2"),
                html.P(f"Progress: {len(answers)} / {MAX_QUESTIONS} answered", className="text-muted")
            ]
    elif step == 3:
        evaluation_content = [
            dbc.Alert("✅ Interview completed! Evaluation is running...", color="success"),
            html.Div(id="evaluation-display"),
        ]
        
    return interview_content, evaluation_content

def trigger_evaluation(step, answers, profile, evaluation, token, candidate_name):
    if step == 3 and evaluation is None and answers:
        eval_result = evaluate_with_fireworks(answers, profile)
        evaluation_json_dump = json.dumps(eval_result, ensure_ascii=False)
        filepath, content = save_results_txt(candidate_name, answers, eval_result, profile)
        mark_interview_completed(token, filepath, evaluation_json_dump, eval_result)
        send_email_notification(candidate_name, os.path.basename(filepath))
        return eval_result, ""
    
    return dash.no_update, dash.no_update

def display_evaluation(evaluation):
    if not evaluation:
        return dbc.Spinner(color="primary")
        
    if "error" in evaluation:
        print(f"Candidate Evaluation Error Details: {evaluation['error']}")
        return dbc.Alert("❌ An error occurred during evaluation. The administrator has been notified.", color="danger")
        
    return dbc.Alert("✅ Thank you. Your interview is complete. Results have been sent to the administrator.", color="success")

def submit_answer(n_clicks, answer_text, questions, current_idx, answers):
    if not n_clicks:
        return answers, current_idx, dash.no_update, dash.no_update
        
    if not answer_text or not answer_text.strip():
        return answers, current_idx, dash.no_update, dbc.Alert("⚠️ Please write an answer before submitting.", color="warning")

    q_text = questions[current_idx]
    answers[q_text] = answer_text.strip()
    
    if current_idx >= MAX_QUESTIONS - 1:
        return answers, current_idx, 3, dbc.Alert("Answer submitted ✅ Starting evaluation...", color="success")
    else:
        return answers, current_idx + 1, 2, dbc.Alert("Answer submitted ✅ Loading next question...", color="success")


def register_callbacks(app):
    @app.callback(
        [
            Output("answers-store", "data"),
            Output("current-index-store", "data"),
            Output("step-store", "data"),
            Output("submit-feedback", "children"),
        ],
        Input("submit-answer-btn", "n_clicks"),
        [
            State("answer-input", "value"),
            State("questions-store", "data"),
            State("current-index-store", "data"),
            State("answers-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def submit_answer(n_clicks, answer_text, questions, current_idx, answers):
        if not n_clicks:
            return answers, current_idx, dash.no_update, dash.no_update

        if not answer_text or not answer_text.strip():
            return (
                answers,
                current_idx,
                no_update,
                dbc.Alert("Warning: Please write an answer before submitting.", color="warning"),
            )

        q_text = questions[current_idx]
        answers = answers.copy()
        answers[q_text] = answer_text.strip()

        if current_idx >= MAX_QUESTIONS - 1:
            return (
                answers,
                current_idx,
                3,
                dbc.Alert("Answer submitted. Starting evaluation...", color="success"),
            )
        else:
            return (
                answers,
                current_idx + 1,
                2,
                dbc.Alert("Answer submitted. Loading next question...", color="success"),
            )
    @app.callback(
        Output("admin-download-result-file", "data"),
        Input({"type": "download-btn", "index": ALL}, "n_clicks"),
        State({"type": "download-btn", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def download_txt_file(n_clicks_list, ids_list):
        if not n_clicks_list or not any(n_clicks_list):
            return no_update

        # which button was clicked?
        for n_clicks, button_id in zip(n_clicks_list, ids_list):
            if n_clicks:
                token = button_id["index"]
                break
        else:
            return no_update

        # DB lookup
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT candidate_name, result_filepath FROM interviews WHERE token = ?",
            (token,),
        )
        row = c.fetchone()
        conn.close()

        if not row or not row[1] or not os.path.exists(row[1]):
            return dcc.send_string(
                f"Error: File not found for token {token}",
                filename="error.txt",
            )

        candidate_name, filepath = row
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return dcc.send_string(content, filename=os.path.basename(filepath))
        except Exception as e:
            print(f"[Download Error] {e}")
            return dcc.send_string(
                f"Error reading file: {e}", filename="download_error.txt"
            )
__all__ = [
    "register_callbacks",
    "interview_layout",
    "admin_panel_layout",
    "init_database",
    "add_new_columns_if_not_exists",
]