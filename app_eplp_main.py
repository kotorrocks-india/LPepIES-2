
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from hashlib import pbkdf2_hmac
from binascii import hexlify, unhexlify

DB_PATH = os.environ.get("EPLP_DB", "eplp_app.db")

# ============================ DB & UTILS ============================

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def execute(sql: str, params: Tuple=()):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()

def df_read(sql: str, params: Tuple=()):
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def ensure_column(conn, table: str, col: str, coltype: str):
    info = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
    cols = set(info["name"].tolist()) if not info.empty else set()
    if col not in cols:
        c = conn.cursor()
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        conn.commit()

def split_title_name(raw_name: str) -> Tuple[str, str]:
    """
    Parse leading Ar./Er. title from name.
    Returns (title, clean_name)
    """
    if not raw_name:
        return "", ""
    s = str(raw_name).strip()
    m = re.match(r"^\s*(ar\.?|er\.?)\s+(.*)$", s, flags=re.IGNORECASE)
    if m:
        prefix = m.group(1).lower().replace(".", "")
        rest = m.group(2).strip()
        title = "Architect" if prefix == "ar" else "Engineer"
        return title, rest
    return "", s

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ============================ INIT DB ============================

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # Core tables
        cur.execute("""CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll TEXT UNIQUE,
            name TEXT,
            year INTEGER,
            degree TEXT,
            email TEXT,
            batch TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT CHECK(type in ('core','visiting')),
            email TEXT,
            title TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS branches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            faculty_id INTEGER,
            FOREIGN KEY(faculty_id) REFERENCES faculty(id)
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS holidays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS faculty_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT,
            faculty_id INTEGER,
            slot INTEGER,
            FOREIGN KEY(faculty_id) REFERENCES faculty(id)
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS subject_criteria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            code TEXT,
            semester INTEGER,
            branch TEXT,
            internal_pass REAL,
            external_pass REAL,
            internal_weight REAL,
            external_weight REAL,
            direct_weight REAL,
            indirect_weight REAL,
            notes TEXT,
            lectures INTEGER,
            studios INTEGER,
            internal_max REAL,
            exam_max REAL,
            jury_max REAL,
            external_total REAL,
            credits REAL,
            objective TEXT,
            syllabus TEXT,
            styles TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS subject_faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sc_id INTEGER,
            faculty_id INTEGER,
            role TEXT CHECK(role in ('sic','lecture','studio')),
            FOREIGN KEY(sc_id) REFERENCES subject_criteria(id),
            FOREIGN KEY(faculty_id) REFERENCES faculty(id)
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sc_id INTEGER,
            title TEXT,
            max_marks REAL,
            due_date TEXT,
            is_exam INTEGER DEFAULT 0,
            is_jury INTEGER DEFAULT 0,
            FOREIGN KEY(sc_id) REFERENCES subject_criteria(id)
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS assignment_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER,
            student_id INTEGER,
            marks REAL,
            FOREIGN KEY(assignment_id) REFERENCES assignments(id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )""")

        # PO/CO + rubrics (placeholders for later)
        cur.execute("""CREATE TABLE IF NOT EXISTS pos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE,
            name TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS subject_cos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sc_id INTEGER NOT NULL,
            co_code TEXT,
            title TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS co_po_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            co_id INTEGER NOT NULL,
            po_id INTEGER NOT NULL,
            corr INTEGER NOT NULL DEFAULT 0
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS assignment_co_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            co_id INTEGER NOT NULL,
            weight REAL NOT NULL DEFAULT 0
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS assignment_rubrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            rubric_no INTEGER NOT NULL,
            name TEXT,
            max_marks REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS assignment_rubric_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            rubric_no INTEGER NOT NULL,
            marks REAL
        )""")

        # Seeds
        if df_read("SELECT COUNT(*) c FROM branches")["c"].iloc[0] == 0:
            for name in ["Humanities", "Design", "Technical", "Allied"]:
                cur.execute("INSERT INTO branches(name) VALUES(?)", (name,))
        if df_read("SELECT COUNT(*) c FROM users")["c"].iloc[0] == 0:
            create_user("superadmin", "superadmin@123", "superadmin", None)
        if df_read("SELECT COUNT(*) c FROM pos")["c"].iloc[0] == 0:
            for i in range(1, 10+1):
                cur.execute("INSERT INTO pos(code,name) VALUES(?,?)", (f"PO{i}", f"Program Outcome {i}"))

        conn.commit()

        # Normalize titles (Ar/Er)
        fac = pd.read_sql_query("SELECT id, name, COALESCE(title,'') as title FROM faculty", conn)
        for _, r in fac.iterrows():
            title, clean = split_title_name(str(r["name"] or ""))
            if title and (not r["title"] or r["title"].strip() == ""):
                cur.execute("UPDATE faculty SET title=?, name=? WHERE id=?", (title, clean, int(r["id"])))
        conn.commit()

# ============================ AUTH ============================

def pbkdf2_hash(password: str, salt: Optional[bytes]=None) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 120000)
    return hexlify(salt).decode() + ":" + hexlify(dk).decode()

def pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = unhexlify(salt_hex.encode())
        dk2 = pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 120000)
        return hexlify(dk2).decode() == dk_hex
    except Exception:
        return False

def create_user(username: str, password: str, role: str, faculty_id: Optional[int]) -> Optional[str]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users(username, password_hash, role, faculty_id) VALUES(?,?,?,?)",
                        (username.lower(), pbkdf2_hash(password), role, faculty_id))
            conn.commit()
        return None
    except Exception as e:
        return str(e)

def set_password(username: str, new_password: str) -> Optional[str]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET password_hash=? WHERE username=?", (pbkdf2_hash(new_password), username.lower()))
            conn.commit()
        return None
    except Exception as e:
        return str(e)

def derive_app_role_for_faculty(faculty_id: int) -> str:
    q = df_read("SELECT role_name FROM faculty_roles WHERE faculty_id=?", (faculty_id,))
    roles = set(q["role_name"].tolist())
    if "principal" in roles: return "principal"
    if "director" in roles: return "director"
    if "branch_head" in roles: return "branch_head"
    if "class_incharge" in roles: return "class_in_charge"
    return "subject_faculty"

def auth_login(username: str, password: str):
    rec = df_read("SELECT username, password_hash, role, faculty_id FROM users WHERE username=?", (username.lower(),))
    if rec.empty:
        return False, "Unknown user"
    row = rec.iloc[0]
    if not pbkdf2_verify(password, row["password_hash"]):
        return False, "Incorrect password"
    role = row["role"]
    faculty_id = row["faculty_id"]
    if role != "superadmin" and pd.notna(faculty_id):
        role = derive_app_role_for_faculty(int(faculty_id))
    return True, {"username": row["username"], "role": role, "faculty_id": int(faculty_id) if pd.notna(faculty_id) else None}

def default_creds_for_name(name: str):
    title, clean = split_title_name(name or "")
    parts = clean.strip().split()
    first = parts[0].lower() if parts else "user"
    last_initial = (parts[-1][0].lower() if len(parts)>1 else "x")
    base = (first[:5] + last_initial)
    return f"{base}1234", f"{base}@1234"

# ============================ BATCH/YEAR HELPERS ============================

PROGRAM_YEARS = 5

def batch_label(join_year: int, years: int = PROGRAM_YEARS) -> str:
    return f"{join_year}-{join_year + years}"

def parse_join_year_from_roll(roll: str) -> Optional[int]:
    if not roll: return None
    m = re.match(r"(\d{4})", str(roll).strip())
    return int(m.group(1)) if m else None

def academic_program_year(join_year: int, today: Optional[date] = None) -> int:
    if today is None:
        today = date.today()
    year_delta = today.year - join_year
    prog = year_delta + 1 if today.month >= 6 else year_delta
    return max(1, min(PROGRAM_YEARS, prog))

def current_year_from_first_roll():
    try:
        df = df_read("SELECT roll FROM students WHERE roll IS NOT NULL AND TRIM(roll)!='' ORDER BY roll LIMIT 1")
        if df.empty: return (None, None)
        roll = str(df["roll"].iloc[0]).strip()
        jy = parse_join_year_from_roll(roll)
        if jy is None: return (roll, None)
        prog_year = academic_program_year(jy)
        return (roll, prog_year)
    except Exception:
        return (None, None)

def current_batch_from_first_roll() -> Optional[str]:
    try:
        df = df_read("SELECT roll FROM students WHERE roll IS NOT NULL AND TRIM(roll)!='' ORDER BY roll LIMIT 1")
        if df.empty: return None
        jy = parse_join_year_from_roll(df["roll"].iloc[0])
        if jy is None: return None
        return batch_label(jy)
    except Exception:
        return None

# ============================ PERMISSIONS ============================

def can_manage_faculty(role: str) -> bool:
    return role in ("superadmin","principal","director")

def can_manage_students(role: str) -> bool:
    return role in ("superadmin","class_in_charge")

def can_manage_holidays(role: str) -> bool:
    return role in ("superadmin","principal","director","class_in_charge")

def can_manage_subjects(role: str) -> bool:
    return role in ("superadmin","principal","director")

# ============================ PAGES ============================

def page_login():
    st.title("EPLP — Sign In")
    with st.form("login_form"):
        u = st.text_input("Username").strip()
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        success, res = auth_login(u, p)
        if not success:
            st.error(str(res))
        else:
            st.session_state["user"] = res
            st.rerun()

def page_students(user: dict):
    role = user.get("role","")
    st.header("Students")
    st.caption("Upload CSV/Excel. Columns: Roll No, Student Name, optional Year, Degree, Email; Batch auto-derived from first roll (e.g., 2022 -> 2022-2027).")

    if can_manage_students(role):
        up = st.file_uploader("Import students (CSV/XLSX/XLS)", type=["csv","xlsx","xls"], key="stu_up")
    else:
        st.info("Read-only. Only **Class In-Charge** or **superadmin** can add/update students.")
        up = None

    join_year_upload = None
    if up and can_manage_students(role):
        try:
            if up.name.lower().endswith(".csv"):
                df_imp = pd.read_csv(up)
            else:
                xls = pd.ExcelFile(up)
                df_imp = xls.parse(xls.sheet_names[0])
            rename = {}
            for c in df_imp.columns:
                cl = str(c).strip().lower()
                if cl in ["roll","roll no","roll number"]: rename[c] = "roll"
                if cl in ["student name","name"]: rename[c] = "name"
                if cl in ["year"]: rename[c] = "year"
                if cl in ["degree"]: rename[c] = "degree"
                if cl in ["email","email id","mail"]: rename[c] = "email"
                if cl in ["batch"]: rename[c] = "batch"
            df_imp = df_imp.rename(columns=rename)

            if not df_imp.empty:
                join_year_upload = parse_join_year_from_roll(df_imp.iloc[0].get("roll"))
            if "year" not in df_imp.columns or df_imp["year"].isna().all():
                if join_year_upload is None:
                    first, y = current_year_from_first_roll()
                    jy = parse_join_year_from_roll(first) if first else None
                    y = academic_program_year(jy) if jy is not None else None
                else:
                    y = academic_program_year(join_year_upload)
                if y is not None:
                    df_imp["year"] = y

            if "batch" not in df_imp.columns or df_imp["batch"].isna().all():
                if join_year_upload is None and not df_imp.empty:
                    join_year_upload = parse_join_year_from_roll(df_imp.iloc[0].get("roll"))
                if join_year_upload is not None:
                    df_imp["batch"] = batch_label(join_year_upload)
                else:
                    b = current_batch_from_first_roll()
                    if b is not None:
                        df_imp["batch"] = b

            for opt in ["degree","email"]:
                if opt not in df_imp.columns:
                    df_imp[opt] = ""

            df_imp = df_imp[["roll","name","year","degree","email","batch"]]
            st.dataframe(df_imp.head(20), use_container_width=True)
            if st.button("Save students", disabled=not can_manage_students(role)):
                with get_conn() as conn:
                    cur = conn.cursor()
                    for _, r in df_imp.iterrows():
                        cur.execute("""INSERT INTO students(roll,name,year,degree,email,batch)
                                       VALUES(?,?,?,?,?,?)
                                       ON CONFLICT(roll) DO UPDATE SET
                                           name=excluded.name, year=excluded.year, degree=excluded.degree, email=excluded.email, batch=excluded.batch
                                    """, (str(r["roll"]).strip(), str(r["name"]).strip(), int(r["year"]) if not pd.isna(r["year"]) else None, str(r.get("degree") or "").strip(), str(r.get("email") or "").strip(), str(r.get("batch") or "").strip()))
                    conn.commit()
                st.success("Saved.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.subheader("All students")
    df_all = df_read("""
        SELECT roll AS 'Roll No', name AS 'Student Name', year AS 'Year', degree AS 'Degree', email AS 'Email', batch AS 'Batch'
        FROM students ORDER BY roll
    """)
    cur_batch = current_batch_from_first_roll()
    if cur_batch:
        st.info(f"Current batch from first roll: **{cur_batch}** (Academic year starts in June).")
    st.dataframe(df_all, use_container_width=True)
    st.download_button("Export students (CSV)", data=df_to_csv_bytes(df_read("SELECT roll,name,year,degree,email,batch FROM students ORDER BY roll")), file_name="students.csv", mime="text/csv")

def page_faculty(user: dict):
    role = user.get("role","")
    st.header("Faculty")
    st.caption("Import core/visiting via CSV/Excel with columns: Name, Type(core|visiting), Email. Names starting with Ar/Er will set title.")
    if can_manage_faculty(role):
        up = st.file_uploader("Import faculty (CSV/XLSX/XLS)", type=["csv","xlsx","xls"], key="fac_up")
    else:
        st.info("Read-only. Only **Principal/Director** or **superadmin** can add/modify faculty and role assignments.")
        up = None

    if up and can_manage_faculty(role):
        try:
            if up.name.lower().endswith(".csv"):
                df_imp = pd.read_csv(up)
            else:
                xls = pd.ExcelFile(up)
                df_imp = xls.parse(xls.sheet_names[0])
            rename = {}
            for c in df_imp.columns:
                cl = str(c).strip().lower()
                if cl in ["name","faculty","faculty name"]: rename[c] = "name"
                if cl in ["type","category"]: rename[c] = "type"
                if cl in ["email","email id"]: rename[c] = "email"
            df_imp = df_imp.rename(columns=rename)
            if "type" not in df_imp.columns: df_imp["type"] = "core"
            df_imp["type"] = df_imp["type"].astype(str).str.lower().map({"core":"core","visiting":"visiting"}).fillna("core")
            with get_conn() as conn:
                cur = conn.cursor()
                for _, r in df_imp.iterrows():
                    raw_name = str(r.get("name") or "").strip()
                    title, clean_name = split_title_name(raw_name)
                    email = str(r.get("email") or "").strip()
                    # skip duplicates (same normalized name+email)
                    exists = pd.read_sql_query(
                        "SELECT id FROM faculty WHERE LOWER(TRIM(name))=? AND LOWER(COALESCE(email,''))=?",
                        conn, params=(clean_name.lower().strip(), email.lower())
                    )
                    if exists.empty:
                        cur.execute("INSERT INTO faculty(name,type,email,title) VALUES(?,?,?,?)",
                                    (clean_name, str(r["type"]).strip(), email, title))
                conn.commit()
            st.success(f"Imported {len(df_imp)} faculty (titles parsed; duplicates skipped).")
        except Exception as e:
            st.error(f"Import failed: {e}")

    fac = df_read("SELECT id, name, COALESCE(title,'') AS title, type, COALESCE(email,'') as email FROM faculty ORDER BY name")
    show = fac.copy()
    show["Display Name"] = show.apply(lambda r: (f"{r['title']} {r['name']}".strip() if r["title"] else r["name"]), axis=1)
    st.dataframe(show[["id","Display Name","type","email","title","name"]], use_container_width=True)
    st.download_button("Export faculty (CSV)", data=df_to_csv_bytes(df_read("SELECT id,name,type,email,title FROM faculty ORDER BY name")), file_name="faculty.csv", mime="text/csv")

    st.subheader("Duplicates (Faculty)")
    dup = df_read("""
        SELECT LOWER(TRIM(name)) as nkey, LOWER(COALESCE(email,'')) as ekey, COUNT(*) c
        FROM faculty
        GROUP BY nkey, ekey
        HAVING COUNT(*)>1
    """)
    if dup.empty:
        st.success("No exact duplicates (same name+email).")
    else:
        st.warning("Exact duplicates detected (same name+email). Merge keeps the oldest ID and relinks references.")
        groups = []
        for _, r in dup.iterrows():
            rows = df_read("SELECT id, COALESCE(title||' ','')||name AS display, type, COALESCE(email,'') email FROM faculty WHERE LOWER(TRIM(name))=? AND LOWER(COALESCE(email,''))=? ORDER BY id", (r['nkey'], r['ekey']))
            groups.append({"Name/Email": f"{rows['display'].iloc[0]} | {rows['email'].iloc[0]}", "IDs": ", ".join(map(str, rows['id'].tolist())), "Count": len(rows)})
        st.dataframe(pd.DataFrame(groups), use_container_width=True)
        if st.button("Merge exact duplicates (keep oldest ID)", disabled=not can_manage_faculty(role)):
            with get_conn() as conn:
                cur = conn.cursor()
                for _, r in dup.iterrows():
                    ids = pd.read_sql_query("SELECT id FROM faculty WHERE LOWER(TRIM(name))=? AND LOWER(COALESCE(email,''))=? ORDER BY id", conn, params=(r['nkey'], r['ekey']))['id'].tolist()
                    keep = ids[0]
                    for rid in ids[1:]:
                        cur.execute("UPDATE users SET faculty_id=? WHERE faculty_id=?", (keep, rid))
                        cur.execute("UPDATE faculty_roles SET faculty_id=? WHERE faculty_id=?", (keep, rid))
                        cur.execute("UPDATE subject_faculty SET faculty_id=? WHERE faculty_id=?", (keep, rid))
                        cur.execute("DELETE FROM faculty WHERE id=?", (rid,))
                conn.commit()
            st.success("Merged duplicates.")
            st.rerun()

    # Role assignments
    st.subheader("Set Principal / Director / Branch Heads / Class In-Charge")
    display_names = ["—"] + show["Display Name"].tolist()
    id_by_display = {show["Display Name"].iloc[i]: int(show["id"].iloc[i]) for i in range(len(show))}

    def pick_role(label, role_name, options=display_names, slot=None):
        sel = st.selectbox(label, options=options, key=f"pick_{role_name}_{slot}")
        if st.button(f"Save {label}", key=f"save_{role_name}_{slot}", disabled=not can_manage_faculty(role)):
            with get_conn() as conn:
                cur = conn.cursor()
                if role_name in ("principal","director"):
                    cur.execute("DELETE FROM faculty_roles WHERE role_name=?", (role_name,))
                    if sel != "—":
                        cur.execute("INSERT INTO faculty_roles(role_name, faculty_id, slot) VALUES(?,?,NULL)", (role_name, id_by_display[sel]))
                elif role_name == "branch_head":
                    cur.execute("DELETE FROM faculty_roles WHERE role_name='branch_head' AND slot=?", (slot,))
                    if sel != "—":
                        cur.execute("INSERT INTO faculty_roles(role_name, faculty_id, slot) VALUES(?,?,?)", ("branch_head", id_by_display[sel], slot))
                elif role_name == "class_incharge":
                    cur.execute("DELETE FROM faculty_roles WHERE role_name='class_incharge' AND slot=?", (slot,))
                    if sel != "—":
                        cur.execute("INSERT INTO faculty_roles(role_name, faculty_id, slot) VALUES(?,?,?)", ("class_incharge", id_by_display[sel], slot))
                conn.commit()
            st.success("Saved. Visible to everyone immediately on refresh.")

    pick_role("Director", "director")
    pick_role("Principal", "principal")

    st.markdown("**Branch Heads**")
    branches = df_read("SELECT * FROM branches ORDER BY name")
    for _, br in branches.iterrows():
        pick_role(f"Head for {br['name']}", "branch_head", options=display_names, slot=int(br["id"]))

    st.markdown("**Class In-Charge (Year 1..5)**")
    for y in range(1,6):
        pick_role(f"Year {y} In-Charge", "class_incharge", options=display_names, slot=y)

def page_holidays(user: dict):
    role = user.get("role","")
    st.header("Holidays")
    if not can_manage_holidays(role):
        st.info("Read-only. Only **Principal/Director/Class In-Charge** or **superadmin** can add holidays.")
    up = st.file_uploader("Import holidays (CSV with columns: date,title)", type=["csv"], key="hol_up") if can_manage_holidays(role) else None
    if up and can_manage_holidays(role):
        try:
            df_imp = pd.read_csv(up)
            rename = {}
            for c in df_imp.columns:
                cl = str(c).strip().lower()
                if cl in ["date"]: rename[c] = "date"
                if cl in ["title","holiday","name"]: rename[c] = "title"
            df_imp = df_imp.rename(columns=rename)
            df_imp = df_imp[["date","title"]]
            with get_conn() as conn:
                cur = conn.cursor()
                for _, r in df_imp.iterrows():
                    cur.execute("INSERT INTO holidays(date,title) VALUES(?,?)", (str(r["date"]), str(r["title"]).strip()))
                conn.commit()
            st.success(f"Imported {len(df_imp)} holidays.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    with st.form("hol_form", clear_on_submit=True):
        d = st.date_input("Holiday date")
        t = st.text_input("Title")
        ok = st.form_submit_button("Add", disabled=not can_manage_holidays(role))
    if ok and can_manage_holidays(role):
        execute("INSERT INTO holidays(date, title) VALUES(?, ?)", (str(d), t.strip()))
        st.success("Added.")

    df_all = df_read("SELECT date AS Date, title AS Holiday FROM holidays ORDER BY date")
    st.dataframe(df_all, use_container_width=True)
    st.download_button("Export holidays (CSV)", data=df_to_csv_bytes(df_read("SELECT date,title FROM holidays ORDER BY date")), file_name="holidays.csv", mime="text/csv")

def page_users_passwords(user: dict):
    st.header("Users & Passwords")
    if user.get("role") != "superadmin":
        st.warning("Only superadmin can manage users here.")
        return

    fac_raw = df_read("SELECT id, name, COALESCE(title,'') AS title, type, COALESCE(email, '') as email FROM faculty ORDER BY id")
    fac_raw["nkey"] = fac_raw["name"].str.strip().str.lower()
    fac_raw["ekey"] = fac_raw["email"].str.strip().str.lower()
    fac = fac_raw.sort_values("id").drop_duplicates(subset=["nkey","ekey"], keep="first")

    _dup = df_read("SELECT COUNT(*) c FROM (SELECT LOWER(TRIM(name)) n, LOWER(COALESCE(email,'')) e, COUNT(*) c FROM faculty GROUP BY n,e HAVING COUNT(*)>1)")
    if not _dup.empty and int(_dup.iloc[0]["c"])>0:
        st.info("Duplicate faculty detected. Use **Faculty → Duplicates (Faculty)** to merge. List below is grouped.")

    users = df_read("SELECT id, username, role, faculty_id FROM users")
    uname_set = set(users["username"].str.lower().tolist()) if not users.empty else set()
    users_by_fac = {}
    if not users.empty:
        for _, u in users.iterrows():
            fid = u["faculty_id"]
            if pd.notna(fid):
                users_by_fac.setdefault(int(fid), []).append(u)

    rows = []
    for _, r in fac.iterrows():
        fid, fname, ftitle = int(r["id"]), str(r["name"]).strip(), str(r["title"]).strip()
        display = f"{ftitle} {fname}".strip() if ftitle else fname
        existing = users_by_fac.get(fid, [])
        if existing:
            primary = sorted(existing, key=lambda x: int(x["id"]))[0]
            rows.append({
                "Faculty": display,
                "Username": primary["username"],
                "Default Password": "(existing)",
                "Derived Role": derive_app_role_for_faculty(fid),
                "Faculty ID": fid,
                "Status": "existing"
            })
        else:
            uname, pwd = default_creds_for_name(display)
            base = uname[:-4]; seq = 1234; candidate = uname
            while candidate.lower() in uname_set:
                seq += 1
                candidate = f"{base}{seq}"
            uname_set.add(candidate.lower())
            rows.append({
                "Faculty": display,
                "Username": candidate,
                "Default Password": pwd,
                "Derived Role": derive_app_role_for_faculty(fid),
                "Faculty ID": fid,
                "Status": "new"
            })

    df_ui = pd.DataFrame(rows)
    st.dataframe(df_ui.drop(columns=["Faculty ID"]), use_container_width=True)

    st.markdown("**Create/Link Users** (sets password to default for NEW; existing rows relink/reset)")
    to_create = st.multiselect("Select faculty to create/link users", df_ui["Faculty"].tolist())
    if st.button("Create/Link selected"):
        count = 0; errs = []
        for fac_name in to_create:
            row = df_ui[df_ui["Faculty"] == fac_name].iloc[0]
            fid = int(row["Faculty ID"])
            status = row["Status"]
            if status == "existing":
                try:
                    _, default_pwd = default_creds_for_name(fac_name)
                    with get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("UPDATE users SET role=?, faculty_id=? WHERE username=?",
                                    (row["Derived Role"], fid, row["Username"].lower()))
                        conn.commit()
                    perr = set_password(row["Username"], default_pwd)
                    if perr: errs.append(f"{fac_name}: {perr}")
                    else: count += 1
                except Exception as e:
                    errs.append(f"{fac_name}: {e}")
            else:
                try:
                    err = create_user(row["Username"], row["Default Password"], row["Derived Role"], fid)
                    if err: errs.append(f"{fac_name}: {err}")
                    else: count += 1
                except Exception as e:
                    errs.append(f"{fac_name}: {e}")
        if errs:
            st.error("Some failed:\n- " + "\n- ".join(errs))
        else:
            st.success(f"Processed {count} faculty.")

    st.divider()
    st.subheader("Duplicate / Orphan Accounts (Admin Tools)")
    st.caption("Duplicates share the same faculty; orphans have no faculty linked.")

    dup = df_read("""
        SELECT faculty_id, COUNT(*) AS cnt
        FROM users
        WHERE faculty_id IS NOT NULL
        GROUP BY faculty_id
        HAVING COUNT(*) > 1
    """)
    if dup.empty:
        st.success("Duplicates by faculty link: none found.")
    else:
        details = []
        for _, r in dup.iterrows():
            fid = int(r["faculty_id"]); cnt = int(r["cnt"])
            names = df_read("SELECT name, COALESCE(title,'') AS title FROM faculty WHERE id=?", (fid,))
            disp = (f"{names['title'].iloc[0]} {names['name'].iloc[0]}").strip() if not names.empty else f"FID {fid}"
            users_list = df_read("SELECT id, username FROM users WHERE faculty_id=? ORDER BY id", (fid,))
            details.append({"Faculty": disp, "Faculty ID": fid, "Count": cnt, "Usernames": ", ".join(users_list["username"].tolist())})
        st.markdown("**Duplicates by Faculty Link**")
        st.dataframe(pd.DataFrame(details), use_container_width=True)
        if st.button("Delete duplicates (keep oldest username per faculty)"):
            removed_total = 0
            with get_conn() as conn:
                cur = conn.cursor()
                for _, r in dup.iterrows():
                    fid = int(r["faculty_id"])
                    rows2 = cur.execute("SELECT id, username FROM users WHERE faculty_id=? ORDER BY id", (fid,)).fetchall()
                    to_delete = [u for (_id, u) in rows2[1:]]
                    if to_delete:
                        cur.executemany("DELETE FROM users WHERE username=?", [(u,) for u in to_delete])
                        removed_total += len(to_delete)
                conn.commit()
            st.success(f"Deleted {removed_total} duplicate user(s).")
            st.rerun()

    orphans = df_read("SELECT id, username, role FROM users WHERE faculty_id IS NULL ORDER BY username")
    if orphans.empty:
        st.info("No orphan users (unlinked accounts).")
    else:
        st.markdown("**Orphan Users (no faculty linked)**")
        st.dataframe(orphans, use_container_width=True)
        fac_display_df = df_read("SELECT id, COALESCE(title||' ','')||name AS disp FROM faculty ORDER BY name")
        fac_display = ["—"] + fac_display_df["disp"].tolist()
        fac_ids = fac_display_df["id"].tolist()
        map_disp_to_id = {"—": None}
        for i, disp in enumerate(fac_display[1:]):
            map_disp_to_id[disp] = int(fac_ids[i])
        pick_users = st.multiselect("Pick orphan usernames to link", options=orphans["username"].tolist())
        pick_fac = st.selectbox("Link selected to faculty", options=fac_display)
        if st.button("Link selected orphans"):
            if pick_fac == "—" or not pick_users:
                st.error("Select at least one username and a faculty.")
            else:
                fid = map_disp_to_id[pick_fac]
                with get_conn() as conn:
                    cur = conn.cursor()
                    cur.executemany("UPDATE users SET faculty_id=? WHERE username=?", [(fid, u) for u in pick_users])
                    conn.commit()
                st.success(f"Linked {len(pick_users)} user(s) to {pick_fac}.")
                st.rerun()

    st.download_button("Export users (CSV)", data=df_to_csv_bytes(df_read("SELECT id,username,role,faculty_id FROM users ORDER BY username")), file_name="users.csv", mime="text/csv")

def page_subject_criteria(user: dict):
    role = user.get("role","")
    st.header("Subject Criteria")
    st.caption("Import/export the subject_criteria table. Only **Principal/Director** or **superadmin** can import/update.")
    up = st.file_uploader("Import subject criteria (CSV)", type=["csv"], key="sc_up") if can_manage_subjects(role) else None
    if up and can_manage_subjects(role):
        try:
            df_imp = pd.read_csv(up)
            rename = {}
            for c in df_imp.columns:
                cl = str(c).strip().lower()
                if cl in ["subject","name"]: rename[c] = "subject"
                if cl in ["code"]: rename[c] = "code"
                if cl in ["semester","sem"]: rename[c] = "semester"
                if cl in ["branch"]: rename[c] = "branch"
                if cl in ["internal_pass","internal passing","internal pass"]: rename[c] = "internal_pass"
                if cl in ["external_pass","external passing","external pass"]: rename[c] = "external_pass"
                if cl in ["internal_weight","internal%","internal percentage"]: rename[c] = "internal_weight"
                if cl in ["external_weight","external%","external percentage"]: rename[c] = "external_weight"
                if cl in ["direct_weight","direct%"]: rename[c] = "direct_weight"
                if cl in ["indirect_weight","indirect%"]: rename[c] = "indirect_weight"
                if cl in ["notes","remark"]: rename[c] = "notes"
                if cl in ["lectures"]: rename[c] = "lectures"
                if cl in ["studios"]: rename[c] = "studios"
                if cl in ["internal_max","internal max"]: rename[c] = "internal_max"
                if cl in ["exam_max","exam max","external marks","exam"]: rename[c] = "exam_max"
                if cl in ["jury_max","jury max","viva","jury"]: rename[c] = "jury_max"
                if cl in ["external_total","external total","external overall"]: rename[c] = "external_total"
                if cl in ["credits","credit"]: rename[c] = "credits"
                if cl in ["objective","subject objective"]: rename[c] = "objective"
                if cl in ["syllabus","course syllabus"]: rename[c] = "syllabus"
                if cl in ["styles","teaching styles"]: rename[c] = "styles"
            df_imp = df_imp.rename(columns=rename)
            needed = ["subject","code","semester"]
            for n in needed:
                if n not in df_imp.columns:
                    df_imp[n] = None
            allowed = ["subject","code","semester","branch","internal_pass","external_pass","internal_weight","external_weight","direct_weight","indirect_weight","notes","lectures","studios","internal_max","exam_max","jury_max","external_total","credits","objective","syllabus","styles"]
            df_imp = df_imp[[c for c in allowed if c in df_imp.columns]]
            with get_conn() as conn:
                cur = conn.cursor()
                for _, r in df_imp.iterrows():
                    code_s = str(r.get("code") or "").strip()
                    sem_v = int(r.get("semester")) if pd.notna(r.get("semester")) and str(r.get("semester")).strip() != "" else None
                    if code_s and sem_v is not None:
                        cur.execute("DELETE FROM subject_criteria WHERE code=? AND semester=?", (code_s, sem_v))
                    cur.execute("""
                        INSERT INTO subject_criteria(subject,code,semester,branch,internal_pass,external_pass,internal_weight,external_weight,direct_weight,indirect_weight,notes,lectures,studios,internal_max,exam_max,jury_max,external_total,credits,objective,syllabus,styles)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        str(r.get("subject") or "").strip(),
                        code_s,
                        sem_v,
                        str(r.get("branch") or "").strip(),
                        float(r.get("internal_pass")) if pd.notna(r.get("internal_pass")) else None,
                        float(r.get("external_pass")) if pd.notna(r.get("external_pass")) else None,
                        float(r.get("internal_weight")) if pd.notna(r.get("internal_weight")) else None,
                        float(r.get("external_weight")) if pd.notna(r.get("external_weight")) else None,
                        float(r.get("direct_weight")) if pd.notna(r.get("direct_weight")) else None,
                        float(r.get("indirect_weight")) if pd.notna(r.get("indirect_weight")) else None,
                        str(r.get("notes") or "").strip(),
                        int(r.get("lectures")) if pd.notna(r.get("lectures")) else None,
                        int(r.get("studios")) if pd.notna(r.get("studios")) else None,
                        float(r.get("internal_max")) if pd.notna(r.get("internal_max")) else None,
                        float(r.get("exam_max")) if pd.notna(r.get("exam_max")) else None,
                        float(r.get("jury_max")) if pd.notna(r.get("jury_max")) else None,
                        float(r.get("external_total")) if pd.notna(r.get("external_total")) else None,
                        float(r.get("credits")) if pd.notna(r.get("credits")) else None,
                        str(r.get("objective") or "").strip(),
                        str(r.get("syllabus") or "").strip(),
                        str(r.get("styles") or "").strip(),
                    ))
                conn.commit()
            st.success(f"Imported {len(df_imp)} subject rows.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    semester = st.selectbox("Semester", options=list(range(1,11)), index=0)
    cb = current_batch_from_first_roll()
    if cb:
        st.caption(f"Batch: {cb} (Sem 1 runs Jun {cb.split('-')[0]} to Apr {int(cb.split('-')[0])+1})")

    df_all = df_read("SELECT * FROM subject_criteria WHERE semester=? ORDER BY subject", (semester,))
    st.dataframe(df_all, use_container_width=True)
    st.download_button("Export subject criteria (CSV)", data=df_to_csv_bytes(df_read("SELECT * FROM subject_criteria ORDER BY semester, code, subject")), file_name="subject_criteria.csv", mime="text/csv")

# ============================ MAIN ============================

def main():
    st.set_page_config(page_title="EPLP", layout="wide")
    init_db()

    user = st.session_state.get("user")
    if not user:
        page_login()
        return

    st.sidebar.write(f"**Logged in as:** {user['username']} ({user['role']})")
    if st.sidebar.button("Sign out"):
        st.session_state.pop("user", None)
        st.rerun()

    pages = {
        "Students": lambda: page_students(user),
        "Faculty": lambda: page_faculty(user),
        "Holidays": lambda: page_holidays(user),
        "Subject Criteria": lambda: page_subject_criteria(user),
        "Users & Passwords": lambda: page_users_passwords(user),
    }
    choice = st.sidebar.selectbox("Go to", options=list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
