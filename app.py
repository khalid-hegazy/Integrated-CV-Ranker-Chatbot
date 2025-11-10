# app.py
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from dash import ALL
load_dotenv()
import src.config as config

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
)
app.title = "HR Team"

try:
    import src.chatbot as chatbot
    from src.chatbot import (
        interview_layout, admin_panel_layout, init_database,
        add_new_columns_if_not_exists, generate_interview_token,
        reset_database, get_all_interviews, validate_interview_token,
        submit_profile, update_interview_display, trigger_evaluation,
        display_evaluation, submit_answer
    )
except Exception as e:
    print(f"Error loading chatbot module: {e}")

custom_styles = {
    "header": {
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "color": "white",
        "padding": "2rem",
        "marginBottom": "2rem",
        "borderRadius": "0 0 20px 20px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
    },
    "card": {
        "borderRadius": "15px",
        "boxShadow": "0 8px 30px rgba(0,0,0,0.1)",
        "border": "none",
        "background": "white",
    },
    "upload_area": {
        "border": "2px dashed #dee2e6",
        "borderRadius": "15px",
        "textAlign": "center",
        "padding": "2rem",
        "margin": "1rem 0",
        "transition": "all 0.3s ease",
        "cursor": "pointer",
    },
    "metric_card": {
        "textAlign": "center",
        "padding": "1.5rem",
        "borderRadius": "15px",
        "background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "color": "white",
        "marginBottom": "1rem",
    },
}

def create_header():
    return dbc.Container(
        [
            html.Div(
                [
                    html.H1(
                        [html.I(className="fas fa-search me-3"), "HR Team"],
                        className="display-4 fw-bold mb-0",
                        style={"color": "#B68648"},
                    ),
                    html.P(
                        "Alandalusia Health Egypt",
                        className="lead mb-0 opacity-75",
                        style={"color": "#B68648"},
                    ),
                ],
                style={
                    "backgroundColor": "#F2F2F2",
                    "backgroundImage": "url('/assets/paper-texture.png')",
                    "backgroundSize": "cover",
                    "backgroundRepeat": "repeat",
                    "padding": "20px",
                    "textAlign": "center",
                },
            )
        ],
        fluid=True,
        className="p-0 mb-4",
    )


def create_upload_component(id_suffix, label, accepted, icon):
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.I(className=f"fas {icon} fa-3x mb-3 text-muted"),
                            html.H5(label, className="card-title"),
                            html.P(
                                f"Drag & drop or click to select {accepted}",
                                className="text-muted small",
                            ),
                            dcc.Upload(
                                id=f"upload-{id_suffix}",
                                children=html.Div(
                                    [html.A("Select File", className="btn btn-outline-primary")]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "100%",
                                    "position": "absolute",
                                    "top": 0,
                                    "left": 0,
                                    "opacity": 0,
                                },
                                multiple=False,
                            ),
                        ],
                        style={**custom_styles["upload_area"], "position": "relative"},
                    )
                ]
            )
        ],
        style=custom_styles["card"],
        className="h-100",
    )


def create_progress_card():
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="fas fa-tasks me-2"), "Processing Status"],
                className="bg-light",
            ),
            dbc.CardBody([html.Div(id="progress-content", children=[html.P("Ready to process files", className="text-muted text-center py-4")])]),
        ],
        style=custom_styles["card"],
        id="progress-card",
        className="mb-4",
    )

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        # Stores ------------------------------------------------------------
        html.Div(
            [
                # Interview system
                dcc.Store(id="questions-store", data=[]),
                dcc.Store(id="answers-store", data={}),
                dcc.Store(id="current-index-store", data=0),
                dcc.Store(id="step-store", data=0),
                dcc.Store(id="token-store", data=""),
                dcc.Store(id="candidate-store", data=""),
                dcc.Store(id="evaluation-store", data=None),
                dcc.Store(id="profile-store", data={}),
                dcc.Store(id="admin-interviews-store", data=[]),
                # CV matcher
                dcc.Store(id="leads-data", storage_type="memory"),
                dcc.Store(id="jd-text", storage_type="memory"),
                dcc.Store(id="processing-results", storage_type="session"),   
                dcc.Store(id="excel-file-path",   storage_type="session"),  
                dcc.Store(id="single-analysis-store", storage_type="session", data=None),
                # Misc
                dcc.Download(id="download-excel"),
                html.Div(id="token-validation-result"),
                html.Div(id="submit-feedback"),
            ]
        ),
    ]
)
init_database()
add_new_columns_if_not_exists()

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname and pathname.startswith("/interview/"):
        return interview_layout()

    # Default – main app with tabs
    return dbc.Container(
        [
            create_header(),
            dbc.Tabs(
                [
                    dbc.Tab(label="Bulk CV Matcher", tab_id="bulk", active_tab_style={"backgroundColor": "#667eea", "color": "white"}),
                    dbc.Tab(label="Single CV Matcher", tab_id="single", active_tab_style={"backgroundColor": "#667eea", "color": "white"}),
                    dbc.Tab(label="Automated Interview", tab_id="admin", active_tab_style={"backgroundColor": "#667eea", "color": "white"}),
                ],
                id="main-tabs",
                active_tab="bulk",
                className="mb-4",
            ),
            html.Div(id="tab-content"),
        ],
        fluid=True,
        className="px-4",
    )

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    [
        State("processing-results", "data"),
        State("excel-file-path", "data"),
        State("single-analysis-store", "data"),
    ],
)
def render_tab_content(active_tab, bulk_res, excel_path, single_ui):
    # ---------- ADMIN ----------
    if active_tab == "admin":
        if not getattr(app, "_db_initialized", False):
            init_database()
            add_new_columns_if_not_exists()
            app._db_initialized = True
        return admin_panel_layout()

    # ---------- BULK ----------
    if active_tab == "bulk":
        base_layout = create_bulk_tab()
        children = list(base_layout.children)

        # Only build results UI if we have BOTH data and path
        results_ui = None
        if bulk_res and excel_path:
            results_ui = _render_bulk_results_ui(bulk_res, excel_path)

        # Replace the placeholder
        for i, child in enumerate(children):
            if getattr(child, "id", None) == "results-content":
                children[i] = html.Div(results_ui or "", id="results-content")
                break

        return html.Div(children)

    # ---------- SINGLE ----------
    if active_tab == "single":
        base_layout = create_single_tab()
        children = list(base_layout.children)

        for i, child in enumerate(children):
            if getattr(child, "id", None) == "single-results":
                children[i] = html.Div(single_ui or "", id="single-results")
                break

        return html.Div(children)

    # fallback
    return create_bulk_tab()

def _render_bulk_results_ui(results_data, excel_path):
    if not results_data or not excel_path:
        return ""

    df = pd.DataFrame(results_data)

    # Metrics
    metrics = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{len(df)}", className="text-white mb-1"),
            html.P("Total CVs Processed", className="mb-0 opacity-75")
        ]), style={**custom_styles["metric_card"],
                   "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"}), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{len(df[df['status']=='Match'])}", className="text-white mb-1"),
            html.P("Matches Found", className="mb-0 opacity-75")
        ]), style={**custom_styles["metric_card"],
                   "background": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"}), md=6),
    ], className="mb-4")

    # Download button
    download_btn = html.Div([
        dbc.Button(
            [html.I(className="fas fa-download me-2"), "Download Excel"],
            id="download-results-btn", color="success", className="me-2"
        )
    ], className="mb-3")

    # Table
    table = dash_table.DataTable(
        id="results-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=results_data,
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"minWidth": "120px", "maxWidth": "250px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        filter_action="native", sort_action="native"
    )

    return dbc.Card([
        dbc.CardHeader([html.I(className="fas fa-chart-bar me-2"), "Analysis Results"], className="bg-light"),
        dbc.CardBody([metrics, download_btn, table])
    ], style=custom_styles["card"], className="mt-4")


def create_bulk_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                create_upload_component("leads", "Upload Leads File", "(.xlsx, .csv)", "fa-file-excel"),
                html.Div(id="leads-preview", className="mt-3"),
            ], md=6),
            dbc.Col([
                create_upload_component("jd-bulk", "Upload Job Description", "(.txt, .pdf, .docx)", "fa-file-text"),
                html.Div(id="jd-preview", className="mt-3"),
            ], md=6),
        ], className="mb-4"),

        dbc.Row([dbc.Col(create_progress_card(), md=12)], className="mb-4"),

        dbc.Row([
            dbc.Col(
                dbc.Button(
                    [html.I(className="fas fa-rocket me-2"), "Start Processing"],
                    id="process-btn", color="primary", size="lg",
                    className="w-100", style={"borderRadius": "15px"}
                ),
                md=6, className="mx-auto"
            )
        ], className="mb-4"),

        # Results will be injected here by render_tab_content
        html.Div(id="results-content")
    ])

def create_single_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                create_upload_component("cv-single", "Upload CV", "(.pdf, .docx)", "fa-file-pdf"),
                html.Div(id="cv-preview"),
            ], md=6),
            dbc.Col([
                create_upload_component("jd-single", "Upload Job Description", "(.txt, .pdf, .docx)", "fa-file-text"),
                html.Div(id="jd-single-preview"),
            ], md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(
                dbc.Button(
                    [html.I(className="fas fa-search me-2"), "Analyze CV"],
                    id="analyze-btn", color="success", size="lg",
                    className="w-100", style={"borderRadius": "15px"}
                ),
                md=6, className="mx-auto"
            )
        ], className="mb-4"),

        # Result will be injected here
        html.Div(id="single-results")
    ])

@app.callback(
    [Output("leads-preview", "children"), Output("leads-data", "data")],
    Input("upload-leads", "contents"),
    State("upload-leads", "filename"),
    prevent_initial_call=True,
)
def update_leads_preview(contents, filename):
    if not contents:
        return "", None
    try:
        _, b64 = contents.split(",")
        decoded = base64.b64decode(b64)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8"))) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(decoded))
        preview = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), f"Uploaded {filename} ({len(df)} rows)"],
            color="success",
            className="mt-3",
        )
        return preview, df.to_dict("records")
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger"), None


@app.callback(
    [Output('jd-preview', 'children'), Output('jd-text', 'data')],
    Input('upload-jd-bulk', 'contents'),
    State('upload-jd-bulk', 'filename')
)
def update_jd_preview(contents, filename):
    if not contents:
        return "", None
    from src.jd_extractor import extract_jd_from_bytes
    try:
        _, b64 = contents.split(",")
        decoded = base64.b64decode(b64)
        jd_text = extract_jd_from_bytes(decoded, filename)
        if not jd_text:
            raise ValueError("Empty JD")
        preview = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), f"Uploaded {filename} ({len(jd_text)} chars)"],
            color="success",
            className="mt-3",
        )
        return preview, jd_text
    except Exception as e:
        return dbc.Alert(f"Error reading JD: {e}", color="danger"), None

@app.callback(
    [
        Output("progress-content", "children"),
        Output("results-content", "children"),
        Output("processing-results", "data"),
        Output("excel-file-path", "data"),
    ],
    Input("process-btn", "n_clicks"),
    [
        State("leads-data", "data"),
        State("jd-text", "data"),
        State("processing-results", "data"),
        State("excel-file-path", "data"),
    ],
)
def process_bulk_analysis(n_clicks, leads_data, jd_text, existing_results, existing_path):
    # ── No click → show stored results (if any) ──
    if n_clicks is None:
        if existing_results and existing_path:
            return (
                dbc.Alert(
                    [html.I(className="fas fa-check-circle me-2"),
                     f"Results loaded from {os.path.basename(existing_path)}"],
                    color="success"
                ),
                _render_bulk_results_ui(existing_results, existing_path),
                existing_results,
                existing_path
            )
        return (
            html.P("Ready to process files", className="text-muted text-center py-4"),
            "", None, None
        )

    # ── Missing files → keep old results or show warning ──
    if not leads_data or not jd_text:
        if existing_results and existing_path:
            return (
                dbc.Alert(
                    [html.I(className="fas fa-exclamation-triangle me-2"),
                     "Please upload both files"],
                    color="warning"
                ),
                _render_bulk_results_ui(existing_results, existing_path),
                existing_results,
                existing_path
            )
        return (
            html.P("Please upload both files", className="text-warning text-center py-4"),
            "", None, None
        )

    # ── REAL PROCESSING ──
    try:
        from src.downloader import download_cvs_from_leads
        from src.ranker import load_cvs_from_dataframe, rank_with_gemini, save_results_to_excel

        # 1. Load leads into DataFrame
        leads_df = pd.DataFrame(leads_data)

        # 2. Save temporarily to download CVs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_csv = f"temp_leads_{timestamp}.csv"
        leads_df.to_csv(temp_csv, index=False)

        # 3. Download CVs
        leads_df = download_cvs_from_leads(
            temp_csv,
            output_dir=f"downloaded_CVs_{timestamp}",
            show_progress=True
        )
        os.remove(temp_csv)

        # 4. Parse CVs
        cvs = load_cvs_from_dataframe(leads_df)

        # 5. Rank with Gemini
        results = rank_with_gemini(
            cvs=cvs,
            job_description=jd_text,
            api_key=config.FIREWORKS_API_KEY or os.getenv("FIREWORKS_API_KEY"),
            batch_size=3,
        )

        # 6. Save to Excel
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        excel_path = os.path.join(out_dir, f"CV_Ranking_Results_{timestamp}.xlsx")
        results_df, _ = save_results_to_excel(results, jd_text, out_dir, excel_path)

        # 7. Build UI
        progress_msg = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"),
             f"Processing complete – saved to {os.path.basename(excel_path)}"],
            color="success"
        )
        results_ui = _render_bulk_results_ui(results_df.to_dict("records"), excel_path)

        return (
            progress_msg,
            results_ui,
            results_df.to_dict("records"),
            excel_path
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            dbc.Alert(
                [html.I(className="fas fa-exclamation-triangle me-2"),
                 f"Error: {str(e)}"],
                color="danger"
            ),
            "", None, None
        )
def render_bulk_results(results_data, excel_path):
    """Helper – returns (progress_msg, layout, data, path)"""
    df = pd.DataFrame(results_data)

    # ---- metrics ----------------------------------------------------
    metrics = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{len(df)}", className="text-white mb-1"),
                            html.P("Total CVs Processed", className="mb-0 opacity-75"),
                        ]
                    ),
                    style={**custom_styles["metric_card"], "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"},
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H3(f"{len(df[df['status']=='Match'])}", className="text-white mb-1"),
                            html.P("Matches Found", className="mb-0 opacity-75"),
                        ]
                    ),
                    style={**custom_styles["metric_card"], "background": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"},
                ),
                md=6,
            ),
        ],
        className="mb-4",
    )

    # ---- download button --------------------------------------------
    download_btn = html.Div(
        dcc.Download(id="download-excel"),
        className="mb-3",
    )

    # ---- table ------------------------------------------------------
    table = dash_table.DataTable(
        id="results-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=results_data,
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"minWidth": "120px", "maxWidth": "250px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        filter_action="native",
        sort_action="native",
    )

    layout = html.Div([metrics, download_btn, table])
    progress = dbc.Alert(
        [html.I(className="fas fa-check-circle me-2"), f"Results loaded from {os.path.basename(excel_path)}"],
        color="success",
    )
    return progress, layout, results_data, excel_path

@app.callback(
    Output("download-excel", "data"),
    Input("download-results-btn", "n_clicks"),
    State("excel-file-path", "data"),
    prevent_initial_call=True,
)
def download_excel_file(n_clicks, excel_path):
    if n_clicks and excel_path and os.path.exists(excel_path):
        return dcc.send_file(excel_path)
    return None

@app.callback(Output("cv-preview", "children"), Input("upload-cv-single", "contents"), State("upload-cv-single", "filename"))
def update_cv_preview(contents, filename):
    if not contents:
        return ""
    return dbc.Alert([html.I(className="fas fa-check-circle me-2"), f"Uploaded {filename}"], color="success", className="py-1 mb-0")


@app.callback(
    [Output("jd-single-preview", "children"), Output("jd-text", "data", allow_duplicate=True)],
    Input("upload-jd-single", "contents"),
    State("upload-jd-single", "filename"),
    prevent_initial_call=True,
)
def update_jd_single_preview(contents, filename):
    if not contents:
        return "", None
    from src.jd_extractor import extract_jd_from_bytes
    try:
        _, b64 = contents.split(",")
        decoded = base64.b64decode(b64)
        jd_text = extract_jd_from_bytes(decoded, filename)
        if not jd_text:
            raise ValueError("Empty JD")
        preview = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), f"Uploaded {filename} ({len(jd_text)} chars)"],
            color="success",
            className="py-1 mb-0",
        )
        return preview, jd_text
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger", className="py-1 mb-0"), None


@app.callback(
    [Output("single-results", "children"), Output("single-analysis-store", "data")],
    Input("analyze-btn", "n_clicks"),
    [
        State("upload-cv-single", "contents"),
        State("upload-cv-single", "filename"),
        State("jd-text", "data"),
        State("single-analysis-store", "data"),
    ],
)
def analyze_single_cv(n_clicks, cv_contents, cv_filename, jd_text, old_ui):
    if n_clicks is None:
        return old_ui or "", old_ui

    if not cv_contents or not jd_text:
        return old_ui or "", old_ui
    if not n_clicks or not cv_contents or not jd_text:
        return "", None

    try:
        from src.ranker import extract_text, rank_with_gemini

        _, b64 = cv_contents.split(",")
        cv_bytes = base64.b64decode(b64)
        tmp_path = f"temp_{cv_filename}"
        with open(tmp_path, "wb") as f:
            f.write(cv_bytes)

        cv_text = extract_text(tmp_path)
        candidate = {"filename": cv_filename, "text": cv_text, "name": cv_filename.split(".")[0], "cv_link": ""}

        results = rank_with_gemini(
            cvs=[candidate],
            job_description=jd_text,
            api_key=config.FIREWORKS_API_KEY or os.getenv("FIREWORKS_API_KEY"),
            batch_size=1,
        )
        os.remove(tmp_path)

        if not results:
            raise ValueError("No result from Gemini")

        r = results[0]
        score = r["score"]
        status = "Match" if score >= 60 else "No Match"
        reasoning = r["reasoning"]

        # gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Match Score"},
                delta={"reference": 60},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 100], "color": "lightblue"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60},
                },
            )
        )
        fig.update_layout(height=300, font={"color": "#2c3e50"})

        card = dbc.Card(
            [
                dbc.CardHeader([html.I(className="fas fa-user-check me-2"), "CV Analysis Result"], className="bg-light"),
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=fig), md=6),
                            dbc.Col(
                                [
                                    html.H4("Match" if score >= 60 else "No Match", className="text-success" if score >= 60 else "text-danger"),
                                    html.Hr(),
                                    html.H6("Analysis Summary:", className="fw-bold"),
                                    html.P(reasoning, className="text-muted"),
                                    html.Hr(),
                                    dbc.Badge(f"Score: {score}/100", color="success" if score >= 60 else "danger", className="fs-6 p-2"),
                                ],
                                className="d-flex flex-column justify-content-center h-100",
                                md=6,
                            ),
                        ]
                    )
                ),
            ],
            style=custom_styles["card"],
            className="mt-4",
        )
        return card, card
    except Exception as e:
        import traceback

        traceback.print_exc()
        return dbc.Alert(f"Error analyzing CV: {e}", color="danger"), None


# ----------------------------------------------------------------------
#  Interview-system callbacks (delegated to src.chatbot)
# ----------------------------------------------------------------------
@app.callback(
    [Output("token-store", "data"), Output("candidate-store", "data"), Output("token-validation-result", "children"),
     Output("questions-store", "data"), Output("step-store", "data")],
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def validate_interview_token_callback(pathname):
    return validate_interview_token(pathname)


@app.callback(
    [Output("profile-store", "data"), Output("questions-store", "data", allow_duplicate=True),
     Output("step-store", "data", allow_duplicate=True), Output("submit-feedback", "children", allow_duplicate=True)],
    Input("submit-profile-btn", "n_clicks"),
    [
        State("full-name-input", "value"),
        State("age-input", "value"),
        State("years-experience-input", "value"),
        State("location-input", "value"),
        State("notice-period-input", "value"),
        State("expected-salary-input", "value"),
        State("specialist-input", "value"),
        State("token-store", "data"),
    ],
    prevent_initial_call=True,
)
def submit_profile_callback(n_clicks, full_name, age, years_experience, location, notice_period, expected_salary, specialist, token):
    return submit_profile(n_clicks, full_name, age, years_experience, location, notice_period, expected_salary, specialist, token)


@app.callback(
    [Output("interview-section", "children"), Output("evaluation-section", "children")],
    [Input("questions-store", "data"), Input("current-index-store", "data"), Input("answers-store", "data"), Input("step-store", "data")],
    prevent_initial_call=True,
)
def update_interview_display_callback(questions, cur_idx, answers, step):
    return update_interview_display(questions, cur_idx, answers, step)


@app.callback(
    [Output("evaluation-store", "data", allow_duplicate=True), Output("submit-feedback", "children", allow_duplicate=True)],
    Input("step-store", "data"),
    [State("answers-store", "data"), State("profile-store", "data"), State("evaluation-store", "data"),
     State("token-store", "data"), State("candidate-store", "data")],
    prevent_initial_call=True,
)
def trigger_evaluation_callback(step, answers, profile, evaluation, token, candidate_name):
    return trigger_evaluation(step, answers, profile, evaluation, token, candidate_name)


@app.callback(Output("evaluation-display", "children"), Input("evaluation-store", "data"))
def display_evaluation_callback(evaluation):
    return display_evaluation(evaluation)


@app.callback(
    [Output("answers-store", "data", allow_duplicate=True), Output("current-index-store", "data", allow_duplicate=True),
     Output("step-store", "data", allow_duplicate=True), Output("submit-feedback", "children", allow_duplicate=True)],
    Input("submit-answer-btn", "n_clicks"),
    [State("answer-input", "value"), State("questions-store", "data"), State("current-index-store", "data"), State("answers-store", "data")],
    prevent_initial_call=True,
)
def submit_answer_callback(n_clicks, answer_text, questions, cur_idx, answers):
    return submit_answer(n_clicks, answer_text, questions, cur_idx, answers)

@app.callback(
    Output("admin-generated-link-output", "children", allow_duplicate=True),
    Input("admin-generate-link-btn", "n_clicks"),
    [
        State("admin-candidate-name", "value"),
        State("admin-candidate-email", "value"),
        State("admin-hours-valid", "value"),
        State("url", "href"),
    ],
    prevent_initial_call=True,
)
def admin_generate_link(n_clicks, name, email, hours, current_url):
    if not n_clicks or not name:
        return no_update

    try:
        hours_valid = int(hours) if hours else 48
        token = generate_interview_token(name, email or "", hours_valid)

        base = current_url.split("/admin")[0].rstrip("/")
        interview_url = f"{base}/interview/{token}"

        return dbc.Alert(
            [
                html.H5("Interview Link Generated!", className="mb-3"),
                html.P(f"Candidate: {name}"),
                html.P(f"Valid for: {hours_valid} h"),
                html.Hr(),
                html.Label("Copy link:"),
                dbc.InputGroup([dbc.Input(value=interview_url, readonly=True)]),
                html.Small(
                    f"Expires: {(datetime.now() + timedelta(hours=hours_valid)).strftime('%Y-%m-%d %H:%M')}",
                    className="text-muted mt-2 d-block",
                ),
            ],
            color="success",
        )
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")


@app.callback(
    Output("admin-interviews-list", "children", allow_duplicate=True),
    [Input("admin-refresh-list-btn", "n_clicks"), Input("admin-generated-link-output", "children"), Input("admin-db-reset-output", "children")],
    prevent_initial_call=True,
)
def update_interviews_list(*_):
    interviews = get_all_interviews()
    if not interviews:
        return dbc.Alert("No interviews yet", color="info")

    rows = []
    for i in interviews:
        (
            name, email, token, status, created, completed, expires,
            filepath, json_data, avg_score, recommendation,
            full_name, age, years_exp, loc, notice, exp_sal, specialist,
        ) = i

        badge = {"pending": "warning", "in_progress": "info", "completed": "success"}.get(status, "secondary")
        rows.append(
            html.Tr(
                [
                    html.Td(name),
                    html.Td(full_name or "-"),
                    html.Td(age or "-"),
                    html.Td(years_exp or "-"),
                    html.Td(loc or "-"),
                    html.Td(exp_sal or "-"),
                    html.Td(specialist or "-"),
                    html.Td(email or "-"),
                    html.Td(dbc.Badge(status.capitalize(), color=badge)),
                    html.Td(completed[:16] if completed else "-"),
                    html.Td(f"{avg_score:.1f}" if avg_score else "-"),
                    html.Td(recommendation or "-"),
                    html.Td(
                        dbc.Button("Download TXT", id={"type": "download-btn", "index": token}, color="info", size="sm")
                        if status == "completed" and filepath
                        else "-"
                    ),
                ]
            )
        )

    return dbc.Table(
        [html.Thead(html.Tr([html.Th(c) for c in ["Admin Name","Full Name","Age","Exp","Loc","Salary","Specialist","Email","Status","Completed","Score","Recommendation","Results"]])),
         html.Tbody(rows)],
        bordered=True,
        hover=True,
        striped=True,
        size="sm",
    )


@app.callback(
    Output("admin-db-reset-output", "children", allow_duplicate=True),
    Input("admin-reset-db-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_db_reset(n_clicks):
    if n_clicks:
        reset_database()
        return dbc.Alert("Database cleared!", color="success")
    return ""

@app.callback(
    Output('download-excel', 'data', allow_duplicate=True),
    Input({'type': 'download-btn', 'index': ALL}, 'n_clicks'),
    State({'type': 'download-btn', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def simple_download_handler(clicks_list, ids_list):
    if not clicks_list or not any(clicks_list):
        return dash.no_update
    
    # Find which button was clicked
    clicked_idx = None
    for i, clicks in enumerate(clicks_list):
        if clicks:
            clicked_idx = i
            break
    
    if clicked_idx is None:
        return dash.no_update
    
    token = ids_list[clicked_idx]['index']
    
    # Get file from database
    import sqlite3
    conn = sqlite3.connect('interviews.db')
    c = conn.cursor()
    c.execute("SELECT result_filepath FROM interviews WHERE token = ?", (token,))
    row = c.fetchone()
    conn.close()
    
    if row and row[0] and os.path.exists(row[0]):
        return dcc.send_file(row[0])
    
    return dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)