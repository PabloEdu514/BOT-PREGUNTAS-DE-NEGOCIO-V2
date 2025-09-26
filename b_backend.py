# b_backend.py
import os
import re
import time
import sqlite3
import requests
import streamlit as st
import pandas as pd
import gdown
import unicodedata

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate

# =========================
# Descarga de la base de datos
# =========================
@st.cache_data(ttl=3600)
def download_database():
    db_path = "ecommerce.db"

    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        if file_size > 1000:
            return db_path
        else:
            os.remove(db_path)

    try:
        file_id = "16SxTNCnJAk2zh9U61gWkm0ggSokqxM89"
        url = f"https://drive.google.com/uc?id={file_id}"

        progress_container = st.container()
        with progress_container:
            st.info("Descargando base de datos... Esto puede tardar unos segundos.")
            progress_bar = st.progress(10)
            status_text = st.empty()

            try:
                status_text.text("Conectando con Google Drive...")
                output = gdown.download(url, db_path, quiet=True)
                if output:
                    progress_bar.progress(100)
                    status_text.text("Base de datos descargada exitosamente.")
                    time.sleep(1)
                    return db_path
                else:
                    raise Exception("gdown no pudo descargar el archivo")
            except Exception:
                status_text.text("Intentando método alternativo...")
                progress_bar.progress(50)

                session = requests.Session()
                response = session.get(f"https://drive.google.com/uc?export=download&id={file_id}", stream=True)

                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                urls = [f"https://drive.google.com/uc?export=download&id={file_id}"]
                if token:
                    urls.insert(0, f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}")

                for url in urls:
                    try:
                        response = session.get(url, stream=True, timeout=300)
                        if response.status_code == 200 and 'text/html' not in response.headers.get('content-type', ''):
                            total_size = int(response.headers.get('content-length', 0))
                            block_size = 8192
                            downloaded = 0
                            temp_path = db_path + ".tmp"

                            with open(temp_path, 'wb') as f:
                                for chunk in response.iter_content(block_size):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total_size > 0:
                                            progress = int(50 + (downloaded / total_size) * 50)
                                            progress_bar.progress(progress)
                                            status_text.text(f"Descargando... {downloaded / 1024 / 1024:.1f} MB")

                            try:
                                conn = sqlite3.connect(temp_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                                cursor.close()
                                conn.close()

                                os.rename(temp_path, db_path)
                                progress_bar.progress(100)
                                status_text.text("Base de datos descargada y verificada.")
                                time.sleep(1)
                                return db_path

                            except sqlite3.DatabaseError:
                                os.remove(temp_path)
                                status_text.text("Archivo descargado no válido (no es SQLite).")
                                continue
                    except:
                        continue

                raise Exception("No se pudo descargar el archivo.")
    except Exception as e:
        st.error(f"Error al descargar base de datos: {str(e)}")
        st.error("Verifica que el archivo sea público en Drive.")
        return None
    finally:
        if 'progress_container' in locals():
            progress_container.empty()

# =========================
# Inicialización DB y API
# =========================
@st.cache_resource
def init_database():
    try:
        db_path = download_database()
        if db_path and os.path.exists(db_path):
            return SQLDatabase.from_uri(f"sqlite:///{db_path}")
    except Exception as e:
        st.error(f"Error al inicializar la base de datos: {e}")
    return None

db = init_database()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    try:
        import a_env_vars
        os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY
    except ImportError:
        st.warning("No se encontró la API Key de OpenAI.")

# =========================
# Seguridad y utilidades
# =========================
# --- Reemplaza tu versión por esta ---
def es_consulta_segura(sql: str) -> bool:
    """
    Permite solo SELECT/CTE y bloquea DDL/DML reales.
    Acepta el uso de funciones como REPLACE() dentro de SELECT.
    """
    s = (sql or "").strip()

    # Quitar comentarios
    s = re.sub(r'--.*?$', '', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    # Normalizar para matching
    slow = re.sub(r'\s+', ' ', s).strip().lower()

    # Debe iniciar con SELECT o WITH (CTE)
    if not (slow.startswith("select") or slow.startswith("with")):
        return False

    # Prohibir múltiples sentencias
    if ';' in slow.strip().rstrip(';'):
        return False

    # Bloquear DDL/DML peligrosos (palabras completas)
    dangerous = [
        r'\binsert\b', r'\bupdate\b', r'\bdelete\b', r'\bdrop\b', r'\balter\b',
        r'\bcreate\b', r'\btruncate\b', r'\battach\b', r'\bdetach\b',
        r'\bpragma\b', r'\bexec(ute)?\b', r'\bvacuum\b',
        r'\bunion\s+select\b'  # evita exfiltración vía UNION
    ]
    if any(re.search(p, slow) for p in dangerous):
        return False

    # Bloquear específicamente "replace into" o "or replace" en DDL/DML,
    # pero permitir la función REPLACE() en SELECT
    if re.search(r'\breplace\s+into\b', slow) or re.search(r'\b(or\s+)?replace\b\s+(into|table|view|index)\b', slow):
        return False

    return True


def quitar_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


# ---------- Normalización visual de IDs ----------
ID_COLS = {"NUMERO SOCIO"}  # agrega más si quieres

def _to_plain_int_str(v):
    if v is None:
        return ""
    s = str(v).strip()
    # 1.01346e+06 -> 1013460
    if re.fullmatch(r'-?\d+(?:\.\d+)?[eE][+-]?\d+', s):
        try:
            return str(int(float(s)))
        except:
            return s
    # 1012345.0 -> 1012345
    if re.fullmatch(r'-?\d+\.0+', s):
        return s.split('.')[0]
    return s

def normalizar_ids_en_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.strip().upper() in ID_COLS:
            df[c] = df[c].map(_to_plain_int_str)
    return df

def desnotacion_cientifica_texto(texto: str) -> str:
    # Post-proceso por si el LLM escribió IDs en notación científica
    def repl(m):
        try:
            return str(int(float(m.group(0))))
        except:
            return m.group(0)
    return re.sub(r'\b-?\d+(?:\.\d+)?[eE][+-]?\d+\b', repl, texto)



# ---------- Preferir LISTADO cuando el usuario pide "tabla/lista" ----------
_LISTA_TRIGGERS = [
    "tabla", "lista", "listado", "listar", "muestrame", "muéstrame",
    "dame", "dime", "enséñame", "numeros de socios", "números de socios",
    "registros", "filas", "exportar", "exportame", "exportáme"
]
_CONTEO_TRIGGERS = ["cuantos", "cuántos", "conteo", "total", "número de", "numero de", "cantidad"]

def usuario_quiere_listado(texto: str) -> bool:
    t = (texto or "").lower()
    return any(w in t for w in _LISTA_TRIGGERS) and not any(w in t for w in _CONTEO_TRIGGERS)

def _col_numero_socio() -> str:
    # Busca la columna "NUMERO SOCIO" según el esquema real
    for c in ALLOWED_COLS:
        if c.strip().upper() == "NUMERO SOCIO":
            return c
    # Fallback seguro
    return "NUMERO SOCIO"

def _extraer_where(sql: str) -> str:
    # Captura el WHERE hasta antes de GROUP/ORDER/HAVING/LIMIT o fin
    m = re.search(r'(?is)\bwhere\b(?P<w>.*?)(?=\bgroup\b|\border\b|\bhaving\b|\blimit\b|$)', sql)
    return f" WHERE {m.group('w').strip()}" if m else ""

def reescribir_count_a_listado(sql: str) -> str:
    where = _extraer_where(sql)
    col = _col_numero_socio()
    return f'SELECT DISTINCT "{col}" FROM socios{where}'

def forzar_listado_si_usuario_pide_tabla(sql: str, pregunta: str) -> str:
    if not sql or not usuario_quiere_listado(pregunta):
        return sql
    if "count(" in sql.lower():
        # Nota UX opcional
        st.session_state["ux_hint"] = "Pediste tabla/lista; convertí un CONTEO a un listado."
        return reescribir_count_a_listado(sql)
    return sql





# ---------- Normalización de REGION y SUCURSAL (robusta) ----------
def _unaccent_upper(s: str) -> str:
    if s is None:
        return ""
    n = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in n if not unicodedata.combining(c)).upper().strip()

def _qualified_col_regex(col: str) -> str:
    """
    Devuelve un patrón de regex que captura la columna con o sin alias:
    - REGION, "REGION"
    - socios.REGION, socios."REGION"
    - "socios".REGION, "socios"."REGION"
    """
    esc = re.escape(col)
    return rf'(?P<qcol>(?:(?:"[^"]+"|[A-Za-z_]\w*)\.)*"?{esc}"?)'

def _norm_eq_in_like(sql: str, col: str) -> str:
    s = sql
    col_re = _qualified_col_regex(col)

    # = 'literal'
    s = re.sub(
        rf'(?is){col_re}\s*=\s*\'(?P<val>[^\']*)\'',
        lambda m: f"UPPER(TRIM({m.group('qcol')})) = '{_unaccent_upper(m.group('val'))}'",
        s
    )

    # IN ('a','b', ...)
    def _repl_in(m):
        items_str = m.group('items')
        lits = re.findall(r"'([^']*)'", items_str)
        if lits:
            items = ", ".join(f"'{_unaccent_upper(x)}'" for x in lits)
        else:
            items = items_str
        return f"UPPER(TRIM({m.group('qcol')})) IN ({items})"

    s = re.sub(
        rf'(?is){col_re}\s+IN\s*\((?P<items>[^)]*)\)',
        _repl_in,
        s
    )

    # LIKE 'literal'
    s = re.sub(
        rf'(?is){col_re}\s+LIKE\s*\'(?P<val>[^\']*)\'',
        lambda m: f"UPPER(TRIM({m.group('qcol')})) LIKE '{_unaccent_upper(m.group('val'))}'",
        s
    )

    return s

def normalizar_region_y_sucursal(sql: str) -> str:
    s = _norm_eq_in_like(sql, "REGION")
    s = _norm_eq_in_like(s, "SUCURSAL")
    # Fallback suave: si quedó alguna igualdad directa sobre REGION, forzar NOCASE
    s = re.sub(
        r'(?is)(?:(?:"[^"]+"|[A-Za-z_]\w*)\.)*"?REGION"?\s*=\s*\'([^\']*)\'(?!\s*collate\s+nocase)',
        lambda m: m.group(0) + " COLLATE NOCASE",
        s
    )
    return s

# (Mantenemos el nombre para compatibilidad; no cambies llamadas existentes)
def corregir_sql_sucursal(sql):
    return _norm_eq_in_like(sql, "SUCURSAL")


def eliminar_limit_si_lista_sucursales(sql):
    if re.search(r'select\s+distinct\s+"?sucursal"?\s+from', sql, re.IGNORECASE):
        return re.sub(r'limit\s+\d+', '', sql, flags=re.IGNORECASE)
    return sql

def dejar_solo_un_statement(sql: str) -> str:
    return sql.split(";")[0].strip()

# ---------- Limit harmonizer ----------
def quitar_limits_global(sql: str) -> str:
    if not sql:
        return sql
    s = sql
    s = re.sub(r'(?is)\blimit\s+\d+(?:\s+offset\s+\d+)?\s*;?', ' ', s)  # LIMIT [n] [OFFSET m]
    s = re.sub(r'(?is)\bfetch\s+first\s+\d+\s+rows\s+only\s*;?', ' ', s)
    s = re.sub(r'(?is)\boffset\s+\d+\s+rows\s+fetch\s+next\s+\d+\s+rows\s+only\s*;?', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

_NUM_WORDS = {
    "un":1, "uno":1, "una":1,
    "dos":2, "tres":3, "cuatro":4, "cinco":5, "seis":6,
    "siete":7, "ocho":8, "nueve":9, "diez":10,
    "once":11, "doce":12, "trece":13, "catorce":14,
    "quince":15, "veinte":20
}

def _palabras_a_digitos(t: str) -> str:
    return re.sub(
        r'\b(un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|veinte)\b',
        lambda m: str(_NUM_WORDS[m.group(1)]), t, flags=re.IGNORECASE
    )

def extraer_limit_de_pregunta(texto: str):
    if not texto:
        return None
    t = texto.lower()
    # Quitar años
    t = re.sub(r'\b20\d{2}\b', ' ', t)
    t = _palabras_a_digitos(t)
    t = re.sub(r'\btop\s*-\s*(\d+)\b', r'top \1', t)
    t = re.sub(r'\btop(\d+)\b', r'top \1', t)

    patrones = [
        r'\btop\s+(\d+)\b',
        r'\bprimer(?:os|as)?\s+(\d+)\b',
        r'\b(dame|dime|mu[eé]strame|ens[eé]ñame|trae|pon|lista|listar|muestra|mostrar|encu[eé]ntrame)\s+(\d{1,4})\b',
        r'\b(\d{1,4})\s+(socios?|clientes?|registros?|filas?|resultados?|n[uú]meros?)\b',
        r'\blos\s+(\d{1,4})\s+m[aá]s\b',
        r'\b(ultim[oa]s?)\s+(\d{1,4})\b',
    ]
    for pat in patrones:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            nums = [int(x) for x in m.groups() if x and str(x).isdigit()]
            if nums:
                n = nums[-1]
                if n > 0:
                    return n
    return None

def _pide_todo(texto: str) -> bool:
    if not texto:
        return False
    return bool(re.search(r'\b(todas?|todo|completa|sin\s+l[ií]mite)\b', texto.lower()))

def agregar_limit_si_no_existe(sql: str, n: int) -> str:
    if not sql:
        return sql
    if re.search(r'\blimit\s+\d+\b', sql, flags=re.IGNORECASE):
        return sql
    return sql.rstrip(';') + f' LIMIT {n}'

# ---------- Memoria simple ----------
def actualizar_ultimo_socio(sql):
    patron = re.compile(r'"?NUMERO SOCIO"?\s*=\s*(\d+)', re.IGNORECASE)
    match = patron.search(sql)
    if match:
        st.session_state["ultimo_socio_consultado"] = int(match.group(1))

def expandir_pregunta_con_memoria(pregunta):
    if not st.session_state.get("ultimo_socio_consultado"):
        return pregunta
    numero = st.session_state["ultimo_socio_consultado"]
    patrones = ["el primero", "el anterior", "este socio", "del primero", "del anterior"]
    for p in patrones:
        if p in pregunta.lower():
            return f"{pregunta} (número de socio {numero})"
    return pregunta

# ---------- Bloqueos de seguridad ----------
SECURITY_BLOCK_TEXT = (
    "🚫 Acción bloqueada por seguridad: este bot solo consulta la base (operaciones SELECT). "
    "No es posible borrar, modificar ni crear datos o estructuras."
)

PATS_DANGEROUS = [
    r"\b(borrar|borra|eliminar|elimina|vaciar|vacía|truncate)\b",
    r"\b(drop\s+(table|database)|alter\s+(table|database)|create\s+(table|database|index))\b",
    r"\b(insertar|insert|actualiza(r)?|update|delete|reemplazar|replace)\b",
    r"\b(attach|detach|pragma|vacuum)\b",
    r"\bformatea(r)?\b",
    r"\bunion\s+select\b",
    r";\s*(drop|truncate|alter|insert|update|delete|create|attach|detach|pragma)"
]

def detectar_bloqueo_texto_usuario(texto: str) -> str | None:
    t = quitar_acentos((texto or "").lower())
    for pat in PATS_DANGEROUS:
        if re.search(pat, t):
            return SECURITY_BLOCK_TEXT
    return None

# ---------- Whitelist esquema ----------
@st.cache_data
def get_schema_whitelist(db_path="ecommerce.db"):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cols = cur.execute("PRAGMA table_info('socios')").fetchall()
        conn.close()
        allowed_table = "socios"
        allowed_cols = [c[1] for c in cols]
        return allowed_table, set(allowed_cols)
    except Exception:
        return "socios", set()

ALLOWED_TABLE, ALLOWED_COLS = get_schema_whitelist("ecommerce.db")

OFFTOPIC_HINTS = {
    "fuera_alcance": (
        "Fuera de alcance: este bot consulta solo la tabla socios. "
        "Si buscas créditos, colocaciones, transacciones u otros módulos, usa el bot correspondiente. "
        "Campos disponibles (parcial): {cols}"
    )
}

def es_en_alcance(pregunta: str) -> bool:
    fuera = [
        "colocacion","colocaciones","prestamo","préstamo","prestamos","préstamos",
        "transaccion","transacción","transacciones","ticket","ivr","pos","tpv",
        "pagos","pago","morosidad","flujo","interes","interés","spread","venta","ventas",
        "facturas","inventario","producto","productos","orden","pedido","pedidos"
    ]
    if any(w in pregunta.lower() for w in fuera):
        if not any(c.lower().replace("_"," ") in pregunta.lower() for c in ALLOWED_COLS):
            return False
    return True

# --- Reemplaza tu valida_sql_whitelist por esta ---
def valida_sql_whitelist(sql: str) -> bool:
    slow = re.sub(r'\s+', ' ', (sql or '')).lower()
    # Solo permite seleccionar desde la tabla socios (con o sin comillas, con alias opcional)
    if " from " in slow:
        if not re.search(r'\bfrom\s+(?:"socios"|socios)\b', slow):
            return False
    # No se permiten JOINs
    if re.search(r'\bjoin\b', slow):
        return False
    return True


# ---------- Filtros UI ----------
def get_campos_socios():
    return sorted(list(ALLOWED_COLS))

def get_distinct_values(colname: str, limit: int = 200):
    if colname not in ALLOWED_COLS:
        return []
    try:
        conn = sqlite3.connect("ecommerce.db")
        cur = conn.cursor()
        q = f'SELECT DISTINCT "{colname}" FROM socios WHERE "{colname}" IS NOT NULL LIMIT {int(limit)}'
        rows = cur.execute(q).fetchall()
        conn.close()
        vals = [r[0] for r in rows if r[0] is not None]
        if colname.lower() == "sucursal":
            vals = sorted({quitar_acentos(str(v)).upper() for v in vals})
        else:
            vals = sorted({str(v) for v in vals})
        return vals
    except Exception:
        return []

def get_sucursal_region_map(normalizar=True):
    try:
        conn = sqlite3.connect("ecommerce.db")
        cur = conn.cursor()
        rows = cur.execute('SELECT DISTINCT "SUCURSAL", "REGION" FROM socios WHERE "SUCURSAL" IS NOT NULL AND "REGION" IS NOT NULL').fetchall()
        conn.close()
        if normalizar:
            out = []
            for suc, reg in rows:
                suc_norm = quitar_acentos(str(suc)).upper()
                reg_str = str(reg)
                out.append((suc_norm, reg_str))
            return sorted(set(out))
        else:
            return sorted(set((str(s), str(r)) for s, r in rows))
    except Exception:
        return []

def get_regiones():
    return get_distinct_values("REGION")

def get_sucursales():
    return get_distinct_values("SUCURSAL")

def get_sucursales_por_region(region: str):
    pares = get_sucursal_region_map(normalizar=True)
    return sorted({s for (s, r) in pares if str(r) == str(region)})

def get_region_de_sucursal(sucursal_norm: str):
    pares = get_sucursal_region_map(normalizar=True)
    for s, r in pares:
        if s == sucursal_norm:
            return r
    return None

def pertenece_sucursal_a_region(sucursal_norm: str, region: str) -> bool:
    r = get_region_de_sucursal(sucursal_norm)
    return (r is not None) and (str(r) == str(region))

# ---------- Contexto de institución ----------
INSTITUCION = {
    "nombre": (st.secrets.get("INSTITUCION_NOMBRE") or "Caja Morelia Valladolid"),
    "acronimo": (st.secrets.get("INSTITUCION_ACRONIMO") or "CMV"),
    "tipo": (st.secrets.get("INSTITUCION_TIPO") or "Cooperativa de ahorro y préstamo"),
}

@st.cache_data(ttl=1800)
def load_institution_txt(path: str | None = None) -> tuple[str, str]:
    ruta = st.secrets.get("INSTITUCION_TXT_PATH") or path or "institucion.txt"
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read().strip(), ruta
    except Exception:
        return "", ruta

@st.cache_data(ttl=900)
def get_metricas_socios_basicas() -> dict:
    try:
        conn = sqlite3.connect("ecommerce.db")
        cur = conn.cursor()
        total = cur.execute('SELECT COUNT(*) FROM socios').fetchone()[0] or 0
        regiones = cur.execute('SELECT COUNT(DISTINCT "REGION") FROM socios WHERE "REGION" IS NOT NULL').fetchone()[0] or 0
        sucursales = cur.execute('SELECT COUNT(DISTINCT "SUCURSAL") FROM socios WHERE "SUCURSAL" IS NOT NULL').fetchone()[0] or 0
        conn.close()
        return {"total_socios": int(total), "regiones": int(regiones), "sucursales": int(sucursales)}
    except Exception:
        return {"total_socios": None, "regiones": None, "sucursales": None}

def es_pregunta_contexto(texto: str) -> bool:
    t = (texto or "").lower()
    claves = [
        "institución","institucion","en qué institución","en que institucion","dónde estamos","donde estamos",
        "qué empresa","que empresa","qué cooperativa","que cooperativa",
        "informacion de la cooperativa","información de la cooperativa","sobre la cooperativa",
        "acerca de la cooperativa","info de la cooperativa","datos de la cooperativa",
        "quiénes somos","quienes somos",
        "nombre de la institución","nombre de la institucion",
        "caja morelia","morelia valladolid","cmv",
        "misión","mision","visión","vision","acerca de","about us"
    ]
    return any(k in t for k in claves)

def responder_contexto_desde_txt(pregunta: str) -> str:
    ctx, ruta = load_institution_txt()
    m = get_metricas_socios_basicas()
    nombre = INSTITUCION.get("nombre",""); acr = INSTITUCION.get("acronimo",""); tipo = INSTITUCION.get("tipo","")

    if ctx:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        prompt = (
            "Eres un asistente de {nombre} ({acr}). Responde en español, breve y factual.\n"
            "SOLO usa el siguiente CONTEXTO. Si la respuesta no está en el contexto, responde: "
            "\"No está en el contexto.\".\n\n"
            "### CONTEXTO (desde {ruta}):\n{ctx}\n\n"
            "### PREGUNTA:\n{q}\n\n"
            "### RESPUESTA:"
        ).format(nombre=nombre, acr=acr, ruta=ruta, ctx=ctx[:28000], q=pregunta)
        return llm.invoke(prompt).content.strip()

    def fmt(n):
        try: return f"{int(n):,}"
        except: return "N/D"

    extra = ""
    if m["total_socios"] is not None:
        extra = f" Actualmente contamos con {fmt(m['total_socios'])} socios, en {fmt(m['regiones'])} regiones y {fmt(m['sucursales'])} sucursales."
    return f"Estamos en {nombre} ({acr}), una {tipo}.{extra}"

# =========================
# Inicialización de la cadena LLM
# =========================
@st.cache_resource
def init_chain():
    global db
    if db is None:
        db = init_database()
        if db is None:
            return None, None, None, None
    try:
        llm = ChatOpenAI(model_name='gpt-4', temperature=0)
        query_chain = create_sql_query_chain(llm, db)

        answer_prompt = PromptTemplate.from_template(
            """Eres un analista que responde SOLO con base en la tabla socios.
Si la pregunta no se puede resolver con columnas de socios, responde en una línea:
"Fuera de alcance: este bot solo consulta la tabla socios (campos disponibles)."

Reglas:
- NO inventes columnas ni tablas.
- Si la pregunta menciona sucursal, usa SIEMPRE: UPPER(SUCURSAL)='NOMBRE SIN ACENTO'.
- Si no hay filtro de sucursal y el usuario pide algo sensible (agregados grandes), sugiere agregar sucursal o región.

Tabla: socios
Columnas (parciales): {cols}

Pregunta del usuario: {question}
Consulta SQL generada: {query}
Resultado SQL (primeros registros para contexto): {result}

Respuesta concisa orientada a negocio:"""
        )

        return query_chain, db, answer_prompt, llm
    except Exception as e:
        st.error(f"Error al inicializar la cadena: {str(e)}")
        return None, None, None, None

# =========================
# Casting defensivo en saldos
# =========================
_NUMERIC_TEXT_COLS = [
    'SALDO CUENTA AHORRO',
    'SALDO CUENTA INVERDINAMICA',
    'SALDO'
]

def _cast_expr_for(col_name: str) -> str:
    return f'CAST(REPLACE(REPLACE(REPLACE("{col_name}", ",", ""), "$", ""), " ", "") AS REAL)'

# Casting defensivo en saldos (REEMPLAZA ESTA FUNCIÓN)
def forzar_cast_numerico_en_saldos(sql: str) -> str:
    """
    Reescribe comparaciones numéricas sobre columnas de saldo que podrían ser texto.
    Ej.: "SALDO CUENTA AHORRO" > 500  ->  CAST(... "SALDO CUENTA AHORRO" ...) > 500
    """
    s = sql
    for col in _NUMERIC_TEXT_COLS:
        # Construimos el patrón SIN usar f-string con backslashes en la expresión
        esc = re.escape(col)                 # SALDO\ CUENTA\ AHORRO
        esc_ws = esc.replace(" ", r"\s+")    # permite espacios variables: \s+
        col_pat = '(?P<col>"{}"|{})'.format(esc, esc_ws)

        # Patrón completo: <col> <op> <num>
        pat = col_pat + r'\s*(?P<op>>=|<=|=|>|<)\s*(?P<num>\d+(?:\.\d+)?)'

        def repl(m):
            op = m.group("op")
            num = m.group("num")
            # Usamos el nombre oficial de la columna para el CAST
            return f'{_cast_expr_for(col)} {op} {num}'

        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s
# =========================
# Ejecución de consultas SQL
# =========================
def consulta(pregunta_usuario, debug: bool = False):
    try:
        if "OPENAI_API_KEY" not in os.environ:
            return "❌ No se configuró la API Key.", None, None

        if not es_en_alcance(pregunta_usuario):
            hint = OFFTOPIC_HINTS["fuera_alcance"].format(
                cols=", ".join(get_campos_socios()[:18]) + ("…" if len(ALLOWED_COLS) > 18 else "")
            )
            return hint, None, None

        query_chain, db_sql, prompt, llm = init_chain()
        if not query_chain or not db_sql:
            return "⚠️ No se pudo inicializar el sistema.", None, None

        # Memoria de "último socio"
        pregunta_usuario = expandir_pregunta_con_memoria(pregunta_usuario)

        # 1) Detectar límite pedido por el usuario (o 'todo')
        user_limit = extraer_limit_de_pregunta(pregunta_usuario)
        if _pide_todo(pregunta_usuario):
            user_limit = None

        # 2) Pedir SQL al modelo
        with st.spinner("Generando consulta SQL..."):
            sql_model_raw = query_chain.invoke({"question": pregunta_usuario})

        # 3) Normalizaciones/correcciones (ANTES del harmonizer de LIMIT)
        sql_corr = sql_model_raw
        sql_corr = normalizar_region_y_sucursal(sql_corr)    # REGION y SUCURSAL -> UPPER(TRIM)
        # Fallback opcional: si pegaste corregir_sucursal_inexacta_o_region
        if 'corregir_sucursal_inexacta_o_region' in globals():
            try:
                sql_corr = corregir_sucursal_inexacta_o_region(sql_corr)
            except Exception:
                pass
        sql_corr = forzar_cast_numerico_en_saldos(sql_corr)  # CAST defensivo en saldos
        sql_corr = forzar_listado_si_usuario_pide_tabla(sql_corr, pregunta_usuario)
        sql_corr = eliminar_limit_si_lista_sucursales(sql_corr)
        sql_corr = dejar_solo_un_statement(sql_corr)
        sql_sin_limits = quitar_limits_global(sql_corr)

        # 4) Harmonizer de LIMIT
        if user_limit is not None:
            sql_efectivo = agregar_limit_si_no_existe(sql_sin_limits, user_limit)
            modo = "sql_limit"
        else:
            sql_efectivo = sql_sin_limits
            modo = "sql_no_limit"

        # 5) Validaciones de seguridad
        if not es_consulta_segura(sql_efectivo):
            return SECURITY_BLOCK_TEXT, None, None
        if not valida_sql_whitelist(sql_efectivo):
            return "🔒 La consulta hace referencia a objetos fuera de la tabla socios.", None, None

        # 6) Ejecutar EXACTAMENTE el sql_efectivo
        with st.spinner("Ejecutando consulta segura..."):
            conn = sqlite3.connect("ecommerce.db")
            cur = conn.cursor()
            cur.execute(sql_efectivo)
            columnas = [d[0] for d in cur.description]
            filas = cur.fetchall()
            conn.close()

        # 7) Cinturón y tirantes
        if user_limit is not None and len(filas) > user_limit:
            filas = filas[:user_limit]
            modo += "+client_cap"

        actualizar_ultimo_socio(sql_efectivo)

        # 7.1) DataFrame y normalización de IDs (si está disponible)
        df = pd.DataFrame(filas, columns=columnas)
        _norm_ids_fn = globals().get("normalizar_ids_en_df")
        if callable(_norm_ids_fn):
            try:
                df = _norm_ids_fn(df)
            except Exception:
                pass

        # 8) Resultado para contexto (preview) usando el DF ya normalizado
        try:
            preview_df = df.head(10)
            resultado_ctx = preview_df.to_markdown(index=False)
        except Exception:
            resultado_ctx = " | ".join(df.columns) + "\n"
            resultado_ctx += "\n".join([" | ".join(map(str, f)) for f in df.head(10).values.tolist()])
            if len(df) > 10:
                resultado_ctx += "\n..."

        # 9) Redacción con el MISMO SQL EFECTIVO
        with st.spinner("Generando respuesta..."):
            respuesta = llm.invoke(prompt.format_prompt(
                question=pregunta_usuario,
                query=sql_efectivo,
                result=resultado_ctx,
                cols=", ".join(get_campos_socios()[:25]) + ("…" if len(ALLOWED_COLS) > 25 else "")
            ).to_string())

        # Post-proceso: quitar notación científica en el texto final y añadir nota UX si existe
        texto_resp = respuesta.content
        _desci_fn = globals().get("desnotacion_cientifica_texto")
        if callable(_desci_fn):
            try:
                texto_resp = _desci_fn(texto_resp)
            except Exception:
                pass
        nota = st.session_state.pop("ux_hint", None)
        if nota:
            texto_resp += f"\n\nℹ️ {nota}"

        # 10) DEBUG UI
        if debug:
            with st.expander("🔧 Debug de consulta", expanded=False):
                st.write("**Pregunta usuario:**", pregunta_usuario)
                st.write("**Límite detectado:**", user_limit)
                st.write("**SQL del modelo (raw):**")
                st.code(str(sql_model_raw), language="sql")
                st.write("**SQL sin límites (post-correcciones):**")
                st.code(sql_sin_limits, language="sql")
                st.write("**SQL EFECTIVO ejecutado:**")
                st.code(sql_efectivo, language="sql")
                st.write("**Modo de ejecución:**", modo)
                st.write("**Filas retornadas:**", len(df))
                st.dataframe(df.head(min(10, len(df))))

        return texto_resp, df, sql_efectivo

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None
