import psycopg

conn = psycopg.connect("postgresql://rpm_user:replace_with_a_strong_password@localhost:5432/remote_monitor")
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM patients;")
    print("patients:", cur.fetchone()[0])
conn.close()
