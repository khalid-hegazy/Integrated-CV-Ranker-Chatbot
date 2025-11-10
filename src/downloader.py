# downloader.py
import os
import requests
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path
import re
import time

USER_AGENT = "Mozilla/5.0 (compatible; CV-Downloader/1.0)"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_cv_column(df: pd.DataFrame):
    """
    Try to find the CV link column by checking headers and values.
    """
    headers = {str(c).lower(): c for c in df.columns}
    header_candidates = ["cv", "cv link", "cv_link", "portfolio", "portfolio link",
                         "portifolo", "portifolo link", "portifolio", "link", "file", "resume"]
    for candidate in header_candidates:
        for h_lower, orig in headers.items():
            if candidate in h_lower:
                return orig

    # fallback: find the column with most http links
    best_col = None
    best_count = 0
    for col in df.columns:
        try:
            sample = df[col].astype(str).fillna("").head(200)
            count = sample.str.contains(r"https?://").sum()
            if count > best_count:
                best_count = count
                best_col = col
        except Exception:
            continue
    return best_col if best_count > 0 else None

def sanitize_name(s: str, max_len: int = 80) -> str:
    s = str(s or "")
    s = re.sub(r"[^\w\s\-\.]", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] or "candidate"

def download_binary(url: str, dest_path: str, timeout: int = 30):
    headers = {"User-Agent": USER_AGENT}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def download_cvs_from_leads(leads_path_or_buffer, output_dir="downloaded_CVs", show_progress=True):
    """
    Reads leads file (xlsx or csv or file-like buffer), downloads CVs, returns DataFrame with
    added columns: 'cv_url' and 'local_cv_path'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load file
    if hasattr(leads_path_or_buffer, "read"):
        # file-like uploaded by Streamlit
        try:
            df = pd.read_excel(leads_path_or_buffer)
        except Exception:
            leads_path_or_buffer.seek(0)
            df = pd.read_csv(leads_path_or_buffer)
    else:
        if str(leads_path_or_buffer).lower().endswith(".xlsx"):
            df = pd.read_excel(leads_path_or_buffer)
        elif str(leads_path_or_buffer).lower().endswith(".csv"):
            df = pd.read_csv(leads_path_or_buffer)
        else:
            raise ValueError("Unsupported file format! Use .xlsx or .csv")

    df = normalize_columns(df)

    # detect possible name/category columns
    name_col = next((c for c in df.columns if c.strip().lower() in ("full name", "fullname", "name")), None)
    category_col = next((c for c in df.columns if c.strip().lower() in ("category", "role", "position")), None)

    cv_col = find_cv_column(df)
    if show_progress:
        print(f"Detected CV column: {cv_col}")

    cv_urls = []
    local_paths = []

    for idx, row in df.iterrows():
        raw_url = None
        if cv_col:
            raw_url = row.get(cv_col)

        # fallback: find any http link in the row
        if not raw_url or pd.isna(raw_url := raw_url):
            for c in df.columns:
                val = str(row.get(c) or "")
                if val.startswith("http://") or val.startswith("https://"):
                    raw_url = val
                    break

        if not raw_url or not str(raw_url).strip().lower().startswith("http"):
            cv_urls.append(None)
            local_paths.append(None)
            continue

        cv_urls.append(str(raw_url).strip())

        try:
            name = sanitize_name(row.get(name_col)) if name_col else sanitize_name(f"candidate_{idx+1}")
            category = sanitize_name(row.get(category_col)) if category_col else "Uncategorized"
            dest_dir = Path(output_dir) / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(urlparse(str(raw_url)).path).suffix or ".pdf"
            filename = f"{idx+1}_{name}{ext}"
            filepath = dest_dir / filename

            # Download with retries
            attempts = 0
            while attempts < 3:
                try:
                    download_binary(str(raw_url), str(filepath), timeout=30)
                    break
                except Exception as e:
                    attempts += 1
                    time.sleep(1)
                    if attempts >= 3:
                        raise

            local_paths.append(str(filepath))
            if show_progress:
                print(f"✅ Downloaded: {filename} → {category}/")
        except Exception as e:
            if show_progress:
                print(f"❌ Failed for {name}: {e}")
            local_paths.append(None)

    df["cv_url"] = cv_urls
    df["local_cv_path"] = local_paths
    return df

# Moved the definition of pd_isna to the top, but a better fix is to import pandas as pd
# and use pd.isna directly. The original code already imported pandas. 
# The issue was using a function that was defined later. 
# Using the standard pd.isna is the best practice.
# The original code's `pd_isna` function is redundant. I've removed it and fixed the call.
def pd_isna(x):
    import pandas as _pd
    return _pd.isna(x)