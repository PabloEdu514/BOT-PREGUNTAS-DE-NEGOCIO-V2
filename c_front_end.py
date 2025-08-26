import time
import streamlit as st
import b_backend
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

st.set_page_config(page_title="BOT Socios | Análisis SQL", page_icon="🤖")
st.title("🤖 BOT para contestar PREGUNTAS DE NEGOCIO de la tabla de socios")

st.write("")  # Línea en blanco

st.write("💡 **Ejemplos de consultas útiles:**")

ejemplos = [
    "💰 MUÉSTRAME LOS 5 NÚMEROS DE SOCIOS CON MAYOR SALDO EN DPFs",
    "💳 ¿CUÁNTOS SOCIOS TIENEN TARJETA DE CRÉDITO EN LA REGIÓN ORIENTE?",
    "📊 DAME LA SUMA DE SALDO DE AHORRO DE SOCIOS QUE ESTÁN EN CARTERA VENCIDA",
    "🌎 AGRÚPAME LAS SUMAS DE RESPONSABILIDAD TOTAL DE LOS CRÉDITOS ACTIVOS POR REGIONES",
    "⭐ ¿QUIÉN ES EL SOCIO QUE TIENE EL MAYOR BC SCORE?",
    "🔍 ENCUENTRA 3 REGISTROS DE SOCIOS QUE PERTENEZCAN A SUCURSAL CENTRO QUE NO TENGAN TARJETA DE CRÉDITO Y QUE TENGAN SCORE MAYOR A 700; MUÉSTRAME EL RESULTADO CON LAS COLUMNAS NÚMERO DE SOCIO Y SCORE"
]

for ejemplo in ejemplos:
    st.write(f"- {ejemplo}")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "rewriter_llm" not in st.session_state:
    st.session_state.rewriter_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

REWRITE_PROMPT = PromptTemplate.from_template(
    """Eres un reescritor de preguntas de análisis de datos.
Reescribe la pregunta para que sea totalmente autónoma y completa si depende del historial.

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
        return (out.content or "").strip() or pregunta
    except Exception:
        return pregunta

def _altura_para_df(df_len, max_height=420):
    return min(max_height, 42 + (32 * max(df_len, 1)))

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

if prompt := st.chat_input("¿En qué te puedo ayudar?"):
    st.session_state.mensajes.append({"role": "user", "content": prompt, "df": None})
    with st.chat_message("user"):
        st.write(prompt)

    pregunta_final = reescribir_pregunta_si_aplica(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            texto, df, sql = b_backend.consulta(pregunta_final)

        if "bloqueada por seguridad" in texto.lower():
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

if st.button("🧹Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()
