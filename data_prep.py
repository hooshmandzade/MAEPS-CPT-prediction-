from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import io
import psycopg2
import re

#!pip install gdown -q
#!gdown --id 13NHW7x3PfT4URc8x0XSK9oZdNoebdrxS --output vw_cpt_brussels_params_completeset_20250318_remapped.parquet
raw = pd.read_parquet('vw_cpt_brussels_params_completeset_20250318_remapped.parquet')
raw = raw.drop(['index', 'sondeernummer', 'pkey_sondering'], axis=1)

# database creation 

# !pip -q install psycopg2-binary pandas python-dotenv

# ---------- 1) Connection (Supabase pooled port 6543 + SSL) ----------
user     = "postgres.vsqvzufrqjpjyulhetwl"
password = "MAEPSmaeps12345"
host     = "aws-1-eu-central-1.pooler.supabase.com"
port     = 6543
dbname   = "postgres"

def get_conn():
    return psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
        sslmode="require"
    )

TABLE_STAGING = "stg_cpt_raw"

# ---------- 2) Sanitize columns (you already did this) ----------
def sanitize(name: str) -> str:
    s = re.sub(r'\s+', '_', str(name).strip())
    s = re.sub(r'[^0-9a-zA-Z_]', '_', s)
    if s and s[0].isdigit():
        s = '_' + s
    return s.lower()

raw = raw.copy()
orig2san, seen, new_cols = {}, set(), []
for c in raw.columns:
    base = sanitize(c)
    name = base
    i = 1
    while name in seen:
        name = f"{base}_{i}"; i += 1
    seen.add(name)
    orig2san[c] = name
    new_cols.append(name)
raw.columns = new_cols
print("Columns ->", raw.columns.tolist())

# ---------- 3) Create staging table ----------
ddl_cols = ",\n  ".join([f'"{c}" TEXT' for c in raw.columns])
ddl = f'CREATE TABLE IF NOT EXISTS {TABLE_STAGING} (\n  {ddl_cols}\n);'

with get_conn() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(ddl)
print("Staging table ready.")


# ---------- 4) Bulk load DataFrame via COPY FROM STDIN ----------
cols_quoted = ", ".join([f'"{c}"' for c in raw.columns])
copy_sql = f"""
COPY {TABLE_STAGING} ({cols_quoted})
FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '\"', NULL '');
"""

def copy_df(df_chunk: pd.DataFrame):
    buf = io.StringIO()
    df_chunk.to_csv(buf, index=False)  # header included
    buf.seek(0)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buf)
        conn.commit()

copy_df(raw)
print("Loaded DataFrame into staging table.")

# ---------- 5) Verify row count ----------
with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_STAGING};")
        print("Rows in staging:", cur.fetchone()[0])

TYPED_TABLE = "cpt_brussels_params"
typed_sql = f"""
DROP TABLE IF EXISTS {TYPED_TABLE};
CREATE TABLE {TYPED_TABLE} AS
SELECT
  COALESCE(NULLIF(sondeernummer, ''), NULL)::text               AS sondeernummer,
  NULLIF(x, '')::double precision                               AS x,
  NULLIF(y, '')::double precision                               AS y,
  NULLIF(diepte_mtaw, '')::double precision                     AS diepte_mtaw,
  NULLIF(qc, '')::double precision                               AS qc,
  NULLIF(fs, '')::double precision                               AS fs,
  NULLIF(rf, '')::double precision                               AS rf
  -- TODO: add the rest of your columns with appropriate casts
FROM {TABLE_STAGING};

CREATE INDEX IF NOT EXISTS idx_{TYPED_TABLE}_sondeer ON {TYPED_TABLE}(sondeernummer);
CREATE INDEX IF NOT EXISTS idx_{TYPED_TABLE}_xy ON {TYPED_TABLE}(x, y);
"""

# Separate labeled and unlabeled data

query = "SELECT * FROM raw.labelled_cpt"

with get_conn() as conn:
    labeled_data = pd.read_sql(query, conn)

print("Data loaded from raw.labelled_cpt")
labeled_data.head(5)

labeled_data = labeled_data.dropna(subset=['lithostrat_id'])

numeric_cols = ['x',
 'y',
 'start_sondering_mtaw',
 'diepte_sondering_tot',
 'diepte',
 'diepte_mtaw',
 'qc',
 'fs',
 'qtn',
 'rf',
 'fr',
 'icn',
 'sbt',
 'ksbt']
labeled_data[numeric_cols] = labeled_data[numeric_cols].apply(pd.to_numeric)

# Add an ID column for lithostrat_id, only what we gathered from meeting (will add if file is uploaded)
layer_id = {'Quartair' : 1001,
'Diest' : 1002,
'Bolderberg' : 1003,
'Sint_Huibrechts_Hern' : 1005,
'Ursel' : 1007,
'Asse' : 1008,
'Wemmel' : 1009,
'Lede' : 1010,
'Brussel' : 1011,
'Merelbeke' : 1012,
'Kwatrecht' : 1013,
'Mont_Panisel' : 1014,
'Aalbeke' : 1015,
'Mons_en_Pevele' : 1016}

# No ID number layers as -1
labeled_data['lithostrat_num'] = labeled_data['lithostrat_id'].map(layer_id).fillna(-1).astype(int)
impt_layers_only = labeled_data[labeled_data['lithostrat_num'] != -1]
dummies = pd.get_dummies(impt_layers_only['lithostrat_id'])
impt_layers_only = pd.concat([impt_layers_only, dummies], axis=1)
impt_layers_only['log_qc'] = np.log(impt_layers_only['qc'])
impt_layers_only['log_fs'] = np.log(impt_layers_only['fs'])
impt_layers_only['log_rf'] = np.log(impt_layers_only['rf'])

# create database table for preprocessed data 
# ---------- 1) Connection (Supabase pooled port 6543 + SSL) ----------

TABLE_STAGING = "labelled_cpt_prepared"

# ---------- 2) Sanitize columns (you already did this) ----------

raw = impt_layers_only.copy()
orig2san, seen, new_cols = {}, set(), []
for c in raw.columns:
    base = sanitize(c)
    name = base
    i = 1
    while name in seen:
        name = f"{base}_{i}"; i += 1
    seen.add(name)
    orig2san[c] = name
    new_cols.append(name)
raw.columns = new_cols
print("Columns ->", raw.columns.tolist())

# ---------- 3) Create staging table ----------
ddl_cols = ",\n  ".join([f'"{c}" TEXT' for c in raw.columns])
ddl = f'CREATE TABLE IF NOT EXISTS raw.{TABLE_STAGING} (\n  {ddl_cols}\n);'

with get_conn() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(ddl)
print("Staging table ready.")


# ---------- 4) Bulk load DataFrame via COPY FROM STDIN ----------
cols_quoted = ", ".join([f'"{c}"' for c in raw.columns])
copy_sql = f"""
COPY raw.{TABLE_STAGING} ({cols_quoted})
FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '\"', NULL '');
"""

copy_df(raw)
print("Loaded DataFrame into staging table.")

# ---------- 5) Verify row count ----------
with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM raw.{TABLE_STAGING};")
        print("Rows in staging:", cur.fetchone()[0])

TYPED_TABLE = "cpt_brussels_params"
typed_sql = f"""
DROP TABLE IF EXISTS {TYPED_TABLE};
CREATE TABLE {TYPED_TABLE} AS
SELECT
  COALESCE(NULLIF(sondeernummer, ''), NULL)::text               AS sondeernummer,
  NULLIF(x, '')::double precision                               AS x,
  NULLIF(y, '')::double precision                               AS y,
  NULLIF(diepte_mtaw, '')::double precision                     AS diepte_mtaw,
  NULLIF(qc, '')::double precision                               AS qc,
  NULLIF(fs, '')::double precision                               AS fs,
  NULLIF(rf, '')::double precision                               AS rf
  -- TODO: add the rest of your columns with appropriate casts
FROM {TABLE_STAGING};

CREATE INDEX IF NOT EXISTS idx_{TYPED_TABLE}_sondeer ON {TYPED_TABLE}(sondeernummer);
CREATE INDEX IF NOT EXISTS idx_{TYPED_TABLE}_xy ON {TYPED_TABLE}(x, y);
"""


