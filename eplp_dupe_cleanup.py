import os, sqlite3

DB = os.environ.get("EPLP_DB", "eplp_app.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

print("Using DB:", DB)

# Collapse duplicate users per faculty_id (keep lowest id)
dupes = cur.execute("""
SELECT faculty_id, COUNT(*) c FROM users WHERE faculty_id IS NOT NULL GROUP BY faculty_id HAVING COUNT(*)>1
""").fetchall()
for fid, c in dupes:
    rows = cur.execute("SELECT id, username FROM users WHERE faculty_id=? ORDER BY id", (fid,)).fetchall()
    keep_id = rows[0][0]
    to_delete = [r[1] for r in rows[1:]]
    if to_delete:
        print(f"Faculty {fid}: keeping user id {keep_id}, deleting {to_delete}")
        cur.executemany("DELETE FROM users WHERE username=?", [(u,) for u in to_delete])

# Dedupe faculty exact name+email duplicates (keep lowest id)
dupes_f = cur.execute("""
SELECT LOWER(TRIM(COALESCE(name,''))) as nkey, COALESCE(email,'') as ekey, COUNT(*) c
FROM faculty GROUP BY nkey, ekey HAVING COUNT(*)>1
""").fetchall()
for nkey, ekey, c in dupes_f:
    rows = cur.execute("""SELECT id FROM faculty WHERE LOWER(TRIM(COALESCE(name,'')))=? AND COALESCE(email,'')=? ORDER BY id""", (nkey, ekey)).fetchall()
    keep = rows[0][0]
    for (rid,) in rows[1:]:
        print(f"Faculty dupes: merging {rid} -> {keep}")
        cur.execute("UPDATE users SET faculty_id=? WHERE faculty_id=?", (keep, rid))
        cur.execute("UPDATE faculty_roles SET faculty_id=? WHERE faculty_id=?", (keep, rid))
        cur.execute("UPDATE subject_faculty SET faculty_id=? WHERE faculty_id=?", (keep, rid))
        cur.execute("DELETE FROM faculty WHERE id=?", (rid,))

conn.commit()
conn.close()
print("Cleanup complete.")
