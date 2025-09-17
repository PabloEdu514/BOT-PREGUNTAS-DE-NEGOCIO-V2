# front_app.py (tu archivo de front; si el tuyo se llama distinto, reemplázalo ahí)
import time
import streamlit as st
import b_backend
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

st.set_page_config(page_title="BOT Socios | Análisis SQL", page_icon="🤖", layout="wide")
st.title("🤖 BOT para contestar PREGUNTAS DE NEGOCIO de la tabla de socios")

st.write("")  # Línea en blanco

# =========================
# Panel guía (NUEVO)
# =========================
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("¿Sobre qué puedo preguntar?")
        st.caption("Este bot consulta **solo** la tabla `socios`. No responde colocaciones, transacciones, ni otros módulos.")

        campos = b_backend.get_campos_socios()
        if campos:
            # Render "chips" simples con markdown
            chips = " ".join([f"`{c}`" for c in campos[:30]])
            st.markdown(f"Campos: {chips}" + (" …" if len(campos) > 30 else ""))
        else:
            st.info("No pude leer columnas de `socios` aún.")

        with st.expander("💡 Ver ejemplos listos"):
            ejemplos = [
                "💰 MUÉSTRAME LOS 5 NÚMEROS DE SOCIOS CON MAYOR SALDO EN DPFs",
                "💳 ¿CUÁNTOS SOCIOS TIENEN TARJETA DE CRÉDITO EN LA REGIÓN ORIENTE?",
                "📊 DAME LA SUMA DE SALDO DE AHORRO DE SOCIOS QUE ESTÁN EN CARTERA VENCIDA",
                "🌎 AGRÚPAME LAS SUMAS DE RESPONSABILIDAD TOTAL DE LOS CRÉDITOS ACTIVOS POR REGIONES",
                "⭐ ¿QUIÉN ES EL SOCIO QUE TIENE EL MAYOR BC SCORE?",
                "🔍 ENCUENTRA 3 REGISTROS DE SOCIOS QUE PERTENEZCAN A SUCURSAL CENTRO QUE NO TENGAN TARJETA DE CRÉDITO Y QUE TENGAN SCORE MAYOR A 700; MUÉSTRAME EL RESULTADO CON LAS COLUMNAS NÚMERO DE SOCIO Y SCORE",
            ]
            for ej in ejemplos:
                st.markdown(f"- {ej}")

    with col2:
        st.subheader("Filtros rápidos")

        # ----------------------------
        # BLOQUE MODIFICADO (dependientes y autocorrección)
        # ----------------------------

        # Catálogos (desde back; SUCURSAL viene normalizada UPPER sin acento)
        regiones_all = b_backend.get_regiones()              # texto región tal cual en DB
        sucursales_all = b_backend.get_sucursales()          # UPPER sin acentos
        _ = b_backend.get_sucursal_region_map()              # precarga/valida mapa (no se usa directo aquí)

        # Estado persistente
        if "sel_region" not in st.session_state:
            st.session_state.sel_region = "—"
        if "sel_sucursal" not in st.session_state:
            st.session_state.sel_sucursal = "—"

        # Opciones dependientes: si hay región elegida, filtra sucursales; si no, todas
        if st.session_state.sel_region != "—":
            sucursales_opts = ["—"] + b_backend.get_sucursales_por_region(st.session_state.sel_region)
        else:
            sucursales_opts = ["—"] + sucursales_all

        # Selects con índices estables
        st.selectbox(
            "REGION (opcional)",
            options=["—"] + regiones_all,
            index=(["—"] + regiones_all).index(st.session_state.sel_region)
                  if st.session_state.sel_region in (["—"] + regiones_all) else 0,
            key="sel_region"
        )

        st.selectbox(
            "SUCURSAL (opcional)",
            options=sucursales_opts,
            index=sucursales_opts.index(st.session_state.sel_sucursal)
                  if st.session_state.sel_sucursal in sucursales_opts else 0,
            key="sel_sucursal"
        )

        # AUTOCORRECCIÓN 1:
        # Si eligieron región y luego una sucursal que no pertenece, ajusta región a la correcta de la sucursal.
        if st.session_state.sel_sucursal != "—" and st.session_state.sel_region != "—":
            if not b_backend.pertenece_sucursal_a_region(st.session_state.sel_sucursal, st.session_state.sel_region):
                region_correcta = b_backend.get_region_de_sucursal(st.session_state.sel_sucursal)
                if region_correcta:
                    st.info(f"🛠️ Ajusté **REGION** a **{region_correcta}** porque la sucursal seleccionada pertenece ahí.")
                    st.session_state.sel_region = region_correcta
                    st.rerun()

        # AUTOCORRECCIÓN 2:
        # Si cambian región y la sucursal ya no coincide, limpia sucursal.
        if st.session_state.sel_region != "—" and st.session_state.sel_sucursal != "—":
            if not b_backend.pertenece_sucursal_a_region(st.session_state.sel_sucursal, st.session_state.sel_region):
                st.warning("La sucursal seleccionada no pertenece a esa región. Se limpiará para evitar resultados vacíos.")
                st.session_state.sel_sucursal = "—"
                st.rerun()

        # Variables locales para usar más abajo (opcional, por legibilidad)
        sel_region = st.session_state.sel_region
        sel_sucursal = st.session_state.sel_sucursal

        auto_inyectar = st.checkbox("Agregar estos filtros a mi pregunta", value=False)

# =========================
# Estado de la conversación
# =========================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "rewriter_llm" not in st.session_state:
    st.session_state.rewriter_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Prompt del rewriter con guardas de alcance (NUEVO)
REWRITE_PROMPT = PromptTemplate.from_template(
    """Eres un reescritor de preguntas para análisis SOLO de la tabla `socios`.
- Si la pregunta está fuera de ese alcance, responde EXACTO: "FUERA_DE_ALCANCE".
- Si depende del historial, hazla autónoma.
- No inventes columnas.

Historial:
{history}

Pregunta nueva:
{question}

Pregunta autónoma:"""
)

def construir_historial_texto(max_turnos=2):
    mensajes = st.session_state.mensajes
    pares = []
    i = len(mensajes) - 1
    while i >= 1 and len(pares) < max_turnos:
        if mensajes[i]["role"] == "assistant" and mensajes[i - 1]["role"] == "user":
            u = mensajes[i - 1]["content"]
            a = mensajes[i]["content"]
            pares.append(f"Usuario: {u}\nAsistente: {a}")
            i -= 2
        else:
            i -= 1
    pares.reverse()
    return "\n\n".join(pares) if pares else "(sin historial)"

def reescribir_pregunta_si_aplica(pregunta):
    try:
        h = construir_historial_texto()
        prompt_text = REWRITE_PROMPT.format(history=h, question=pregunta)
        out = st.session_state.rewriter_llm.invoke(prompt_text)
        out_text = (out.content or "").strip()
        return out_text or pregunta
    except Exception:
        return pregunta

def _altura_para_df(df_len, max_height=420):
    return min(max_height, 42 + (32 * max(df_len, 1)))

# Render historial
for i, m in enumerate(st.session_state.mensajes):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("df") is not None:
            height = _altura_para_df(len(m["df"]))
            st.dataframe(m["df"], use_container_width=True, height=height)
            st.download_button("📥 Exportar este resultado a CSV",
                               m["df"].to_csv(index=False).encode("utf-8"),
                               f"resultado_{i}.csv", mime="text/csv",
                               key=f"dl_hist_{i}")

# Input del usuario
prompt = st.chat_input("¿En qué te puedo ayudar?")
if prompt:
    # Inyectar filtros si aplica (no rompe nada; solo agrega contexto en lenguaje natural)
    user_prompt = prompt
    if auto_inyectar:
        extras = []
        if sel_region != "—":
            extras.append(f"en la región {sel_region}")
        if sel_sucursal != "—":
            extras.append(f"en la sucursal {sel_sucursal}")
        if extras:
            user_prompt = f"{user_prompt} ({', '.join(extras)})"

    st.session_state.mensajes.append({"role": "user", "content": user_prompt, "df": None})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Reescritura (puede responder FUERA_DE_ALCANCE)
    pregunta_final = reescribir_pregunta_si_aplica(user_prompt)
    if pregunta_final == "FUERA_DE_ALCANCE":
        with st.chat_message("assistant"):
            st.warning("Fuera de alcance: este bot consulta SOLO la tabla `socios`. Revisa los campos disponibles arriba.")
        st.session_state.mensajes.append({
            "role": "assistant",
            "content": "Fuera de alcance. Usa los campos de `socios`.",
            "df": None
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                texto, df, sql = b_backend.consulta(pregunta_final)

            if "bloqueada por seguridad" in (texto or "").lower():
                st.error("🔒 Consulta bloqueada por seguridad: solo se permiten operaciones SELECT.")
            else:
                st.write(texto)
                if sql is not None:
                    with st.expander("📄 Ver consulta SQL generada"):
                        st.code(sql, language="sql")

            if df is not None:
                height = _altura_para_df(len(df))
                st.dataframe(df, use_container_width=True, height=height)
                st.download_button("📥 Exportar este resultado a CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   f"resultado_{int(time.time())}.csv", mime="text/csv",
                                   key=f"dl_new_{int(time.time()*1000)}")

        st.session_state.mensajes.append({
            "role": "assistant",
            "content": texto,
            "df": df
        })

# Botón reset
if st.button("🧹 Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()
