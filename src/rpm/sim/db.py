import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL. Put it in .env or your environment.")

@contextmanager
def get_conn():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            yield conn, cur

def upsert_patient(pid: str, name: str) -> None:
    with get_conn() as (conn, cur):
        cur.execute(
            """
            INSERT INTO patients (id, name)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;
            """,
            (pid, name),
        )
        conn.commit()

def insert_vitals(pid: str, hr, rr, temp, bp_sys, bp_dia) -> None:
    with get_conn() as (conn, cur):
        cur.execute(
            """
            INSERT INTO vital_snapshots (patient_id, hr, rr, temp, bp_sys, bp_dia)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (pid, hr, rr, temp, bp_sys, bp_dia),
        )
        conn.commit()

def last_30min_hr(pid: str):
    with get_conn() as (_, cur):
        cur.execute(
            """
            SELECT ts, hr
            FROM vital_snapshots
            WHERE patient_id = %s
              AND ts >= NOW() - INTERVAL '30 minutes'
              AND hr IS NOT NULL
            ORDER BY ts ASC;
            """,
            (pid,),
        )
        return cur.fetchall()
