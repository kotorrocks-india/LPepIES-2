
# EPLP App (Streamlit) — v7.4 (Full Package)

**Roles & Permissions**
- **superadmin**: full access to all pages and actions.
- **principal/director**: can add/modify **faculty**, assign **branch heads** & **class in-charges**, manage **holidays**, manage **subject criteria**.
- **class_in_charge**: can add/import **students** and manage **holidays**.
- **branch_head**, **subject_faculty**: read-only (export allowed).

**Key Features**
- Students: Degree/Email optional; **Batch** auto from first student roll (first 4 digits, June-start). Import CSV/XLSX; Export CSV.
- Faculty: Import CSV/XLSX with **Ar/Er → title**; duplicate protection (same normalized **name+email** skipped); Export CSV.
- Faculty Duplicates tool: **merge exact duplicates** (keeps oldest ID and relinks references).
- Users & Passwords (superadmin): create/link users from faculty list; **delete duplicate users**, **link orphan users**.
- Holidays: add manually, **import/export CSV**.
- Subject Criteria: **import/export CSV** (normalized columns; minimal structure ready to expand).

## Run
```bat
cd /d E:\Downloads\eplp
python -m pip install -r requirements.txt
streamlit run app_eplp_main.py
```
Default first-run login: `superadmin / superadmin@123`
