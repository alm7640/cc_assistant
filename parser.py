# parser.py
# Handles ingestion of PDF, CSV, XLS/XLSX, DOCX statement files
# Normalizes all formats into a standard DataFrame schema:
#   date (datetime), merchant (str), amount (float), raw_merchant (str), source_file (str)

import io
import re
import pandas as pd
from datetime import datetime
from typing import Optional
from merchant_map import normalize_merchant


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_amount(val) -> Optional[float]:
    """Convert various amount formats to a positive float charge, or None."""
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace("$", "").replace(" ", "")
    # Some banks use parentheses for debits: (123.45)
    negative = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True
    try:
        amt = float(s)
    except ValueError:
        return None
    # Some exports use negative for charges, positive for payments
    # We want charges as positive — caller decides which sign convention
    return abs(amt) if not negative else abs(amt)


def _looks_like_payment(merchant: str, amount: float, credit_flag=False) -> bool:
    """Heuristic: is this row a payment/credit rather than a purchase?"""
    if credit_flag:
        return True
    m = merchant.lower()
    payment_keywords = [
        "payment", "thank you", "autopay", "credit", "refund",
        "return", "adjustment", "reward", "cashback", "cash back",
        "transfer", "deposit", "interest charge", "fee waiver",
    ]
    return any(kw in m for kw in payment_keywords)


def _parse_date(val) -> Optional[datetime]:
    """Try multiple date formats."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    s = str(val).strip()
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d-%b-%Y",
        "%b %d, %Y", "%B %d, %Y", "%d/%m/%Y", "%m-%d-%Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV bank exports. Handles many column name variants."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, error_bad_lines=False)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Date column detection
    date_candidates = ["date", "transaction_date", "trans_date", "post_date",
                       "posted_date", "activity_date", "transaction date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if not date_col:
        date_col = next((c for c in df.columns if "date" in c), None)

    # Merchant / description column detection
    desc_candidates = ["description", "merchant", "payee", "name", "merchant_name",
                       "transaction_description", "memo", "details", "narrative"]
    desc_col = next((c for c in desc_candidates if c in df.columns), None)
    if not desc_col:
        desc_col = next((c for c in df.columns if any(k in c for k in ["desc", "merch", "payee", "name"])), None)

    # Amount column detection
    amt_candidates = ["amount", "debit", "charge", "transaction_amount",
                      "debit_amount", "withdrawal", "charged_amount"]
    amt_col = next((c for c in amt_candidates if c in df.columns), None)
    if not amt_col:
        amt_col = next((c for c in df.columns if "amount" in c or "debit" in c), None)

    # Credit column (to detect payments)
    credit_col = next((c for c in df.columns if "credit" in c), None)

    if not all([date_col, desc_col, amt_col]):
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        date = _parse_date(row.get(date_col, ""))
        merchant_raw = str(row.get(desc_col, "")).strip()
        amt = _clean_amount(row.get(amt_col, ""))
        is_credit = credit_col and str(row.get(credit_col, "")).strip() not in ("", "0", "0.00", "nan")

        if date is None or amt is None or amt <= 0:
            continue
        if _looks_like_payment(merchant_raw, amt, is_credit):
            continue

        rows.append({
            "date": date,
            "raw_merchant": merchant_raw,
            "merchant": normalize_merchant(merchant_raw),
            "amount": amt,
            "source_file": filename,
        })

    return pd.DataFrame(rows)


def _parse_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse XLS/XLSX exports — tries each sheet."""
    frames = []
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet, dtype=str)
                df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
                # Reuse CSV logic by converting to CSV bytes
                csv_bytes = df.to_csv(index=False).encode()
                parsed = _parse_csv(csv_bytes, filename)
                if not parsed.empty:
                    frames.append(parsed)
            except Exception:
                continue
    except Exception:
        pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_pdf(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse PDF credit card statements.
    Strategy 1: pdfplumber table extraction (structured)
    Strategy 2: raw text line-by-line regex parsing (fallback)
    """
    import pdfplumber

    rows = []

    # ── Strategy 1: Table extraction ─────────────────────────────────────
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [str(h).strip().lower().replace(" ", "_") if h else "" for h in table[0]]
                    for data_row in table[1:]:
                        if not data_row:
                            continue
                        row_dict = {headers[i]: str(data_row[i]).strip() if data_row[i] else ""
                                    for i in range(min(len(headers), len(data_row)))}
                        # Try to find date, merchant, amount in this row
                        date_val = next((row_dict[k] for k in row_dict if "date" in k and row_dict[k]), None)
                        desc_val = next((row_dict[k] for k in row_dict
                                         if any(x in k for x in ["desc", "merch", "payee", "name"]) and row_dict[k]), None)
                        amt_val = next((row_dict[k] for k in row_dict
                                        if any(x in k for x in ["amount", "debit", "charge"]) and row_dict[k]), None)

                        if not amt_val:
                            # Try last numeric-looking column
                            for k in reversed(list(row_dict.keys())):
                                cleaned = row_dict[k].replace(",", "").replace("$", "").replace("(", "").replace(")", "")
                                try:
                                    float(cleaned)
                                    amt_val = row_dict[k]
                                    break
                                except ValueError:
                                    continue

                        if not desc_val:
                            # Use second column as fallback description
                            vals = list(row_dict.values())
                            desc_val = vals[1] if len(vals) > 1 else ""

                        date = _parse_date(date_val) if date_val else None
                        amt = _clean_amount(amt_val) if amt_val else None
                        merchant_raw = str(desc_val).strip() if desc_val else ""

                        if date is None or amt is None or amt <= 0 or not merchant_raw:
                            continue
                        if _looks_like_payment(merchant_raw, amt):
                            continue

                        rows.append({
                            "date": date,
                            "raw_merchant": merchant_raw,
                            "merchant": normalize_merchant(merchant_raw),
                            "amount": amt,
                            "source_file": filename,
                        })
    except Exception:
        pass

    # ── Strategy 2: Text regex fallback ──────────────────────────────────
    if not rows:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                full_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )

            # Pattern: date  description  amount
            # Covers formats like: 01/15/2024  STARBUCKS #1234  5.75
            pattern = re.compile(
                r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+"
                r"([A-Za-z][^\d\n]{3,50?}?)\s+"
                r"\$?([\d,]+\.\d{2})"
            )
            for match in pattern.finditer(full_text):
                date_str, desc, amt_str = match.groups()
                date = _parse_date(date_str)
                amt = _clean_amount(amt_str)
                merchant_raw = desc.strip()

                if date is None or amt is None or amt <= 0:
                    continue
                if _looks_like_payment(merchant_raw, amt):
                    continue

                rows.append({
                    "date": date,
                    "raw_merchant": merchant_raw,
                    "merchant": normalize_merchant(merchant_raw),
                    "amount": amt,
                    "source_file": filename,
                })
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_docx(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse DOCX — extract text then apply regex like PDF fallback."""
    import docx2txt
    import tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text = docx2txt.process(tmp_path)
    except Exception:
        return pd.DataFrame()
    finally:
        os.unlink(tmp_path)

    rows = []
    pattern = re.compile(
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+"
        r"([A-Za-z][^\d\n]{3,50?}?)\s+"
        r"\$?([\d,]+\.\d{2})"
    )
    for match in pattern.finditer(text):
        date_str, desc, amt_str = match.groups()
        date = _parse_date(date_str)
        amt = _clean_amount(amt_str)
        merchant_raw = desc.strip()

        if date is None or amt is None or amt <= 0:
            continue
        if _looks_like_payment(merchant_raw, amt):
            continue

        rows.append({
            "date": date,
            "raw_merchant": merchant_raw,
            "merchant": normalize_merchant(merchant_raw),
            "amount": amt,
            "source_file": filename,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Accept a Streamlit UploadedFile and return a normalized DataFrame.
    Returns empty DataFrame on failure.
    """
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()
    ext = filename.lower().split(".")[-1]

    if ext == "csv":
        df = _parse_csv(file_bytes, filename)
    elif ext in ("xls", "xlsx"):
        df = _parse_excel(file_bytes, filename)
    elif ext == "pdf":
        df = _parse_pdf(file_bytes, filename)
    elif ext == "docx":
        df = _parse_docx(file_bytes, filename)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # Enforce schema and types
    df = df[["date", "merchant", "raw_merchant", "amount", "source_file"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] > 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def combine_files(uploaded_files) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse and combine multiple uploaded files.
    Returns (combined_df, list_of_warnings).
    """
    frames = []
    warnings = []

    for f in uploaded_files:
        df = parse_uploaded_file(f)
        if df.empty:
            warnings.append(f"⚠️ Could not extract transactions from **{f.name}**. "
                            "Check that it's a valid statement export.")
        else:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), warnings

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: same date + merchant + amount within 1 day
    combined = combined.drop_duplicates(
        subset=["date", "merchant", "amount"], keep="first"
    )
    combined = combined.sort_values("date").reset_index(drop=True)

    # Check for month gaps
    if not combined.empty:
        months = pd.period_range(
            start=combined["date"].min().to_period("M"),
            end=combined["date"].max().to_period("M"),
            freq="M",
        )
        covered = set(combined["date"].dt.to_period("M").unique())
        missing = [str(m) for m in months if m not in covered]
        if missing:
            warnings.append(
                f"📅 Possible gaps detected — no transactions found for: {', '.join(missing)}. "
                "Upload missing statements for more accurate analysis."
            )

    return combined, warnings


def extract_raw_text(file_bytes: bytes, filename: str) -> str:
    """Return extracted plain text for debugging purposes for common file types.
    Returns an empty string on failure.
    """
    ext = filename.lower().split(".")[-1]
    try:
        if ext == "pdf":
            import pdfplumber
            texts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
            return "\n\n--- PAGE BREAK ---\n\n".join(texts)

        if ext in ("docx",):
            import docx2txt
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                text = docx2txt.process(tmp_path) or ""
            except Exception:
                text = ""
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return text

        if ext == "csv":
            try:
                return file_bytes.decode("utf-8", errors="replace")
            except Exception:
                return ""

        if ext in ("xls", "xlsx"):
            try:
                xl = pd.ExcelFile(io.BytesIO(file_bytes))
                parts = []
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet, dtype=str)
                    parts.append(f"-- Sheet: {sheet} --\n" + df.to_csv(index=False))
                return "\n\n".join(parts)
            except Exception:
                return ""

    except Exception:
        return ""

    return ""
