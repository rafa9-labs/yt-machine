from db.connection import get_connection
conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'videos_status_check'")
rows = cur.fetchall()
for r in rows:
    print(dict(r))
cur.close()
conn.close()
