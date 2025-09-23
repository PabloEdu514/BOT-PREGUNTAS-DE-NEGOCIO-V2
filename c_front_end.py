# c_front_end.py
import time
import streamlit as st
import b_backend
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

st.set_page_config(
    page_title="BOT Socios | Análisis SQL",
    page_icon="Valladolid.jpeg",
    layout="wide"
)

st.title("🤖 BOT PARA RESPONDER PREGUNTAS DE NEGOCIO SOBRE SOCIOS")
st.write("")

# Estilos generales
st.markdown("""
<style>
.chip-wrap{ max-width:900px; margin:6px 0 0 0; }
.chip-row{ display:flex; flex-wrap:wrap; gap:10px; }
.chip{
  display:inline-block; padding:8px 12px; border-radius:999px;
  background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15);
  font-size:13px; line-height:1; white-space:nowrap;
}
.chip:hover{ background:rgba(255,255,255,0.10); border-color:rgba(255,255,255,0.25); }

.pager-bar{ display:flex; align-items:center; gap:8px; margin-top:6px; }
.pager-info{ font-size:12px; opacity:.75; }
.pager-spacer{ flex:1; }

.stButton>button{
  padding:6px 10px; border-radius:10px; border:1px solid rgba(255,255,255,0.15);
  background:transparent; color:inherit;
}
.stButton>button:hover{ background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.25); }
.stButton>button:disabled{ opacity:.35; }

.full-expander { margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# Render de campos paginados
def render_campos_paginado_15(campos: list[str], ss_key: str = "campos_socios"):
    if not campos:
        st.info("No pude leer columnas de socios aún.")
        return

    PAGE_SIZE = 18
    pkey = f"{ss_key}_page"
    if pkey not in st.session_state:
        st.session_state[pkey] = 1

    total = len(campos)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state[pkey] = max(1, min(st.session_state[pkey], total_pages))
    cur = st.session_state[pkey]

    chips_ph = st.container()
    bar_ph = st.empty()

    def draw_bar(cur_page: int):
        with bar_ph.container():
            left, right = st.columns([8, 2])
            with left:
                st.markdown(
                    f'<div class="pager-bar"><span class="pager-info">'
                    f'Mostrando {PAGE_SIZE} por página</span>'
                    f'<span class="pager-spacer"></span>'
                    f'<span class="pager-info">Página {cur_page} de {total_pages} · {total} campos</span></div>',
                    unsafe_allow_html=True
                )
            with right:
                cprev, cnext = st.columns([1, 1])
                with cprev:
                    prev_clicked_local = st.button("‹", key=f"{ss_key}_prev_{cur_page}",
                                                   disabled=(cur_page <= 1))
                with cnext:
                    next_clicked_local = st.button("›", key=f"{ss_key}_next_{cur_page}",
                                                   disabled=(cur_page >= total_pages))
        return prev_clicked_local, next_clicked_local

    prev_clicked, next_clicked = draw_bar(cur)

    new_cur = cur
    if prev_clicked:
        new_cur = max(1, cur - 1)
    if next_clicked:
        new_cur = min(total_pages, cur + 1)

    if new_cur != cur:
        st.session_state[pkey] = new_cur
        prev_clicked, next_clicked = draw_bar(new_cur)

    cur = st.session_state[pkey]
    start, end = (cur - 1) * PAGE_SIZE, (cur - 1) * PAGE_SIZE + PAGE_SIZE
    visibles = campos[start:end]

    with chips_ph:
        st.markdown('<div class="chip-wrap"><div class="chip-row">', unsafe_allow_html=True)
        st.markdown("\n".join([f'<span class="chip">{c}</span>' for c in visibles]), unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

# Panel de campos y filtros
with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("¿Sobre qué puedo preguntar?")
        st.caption("Este bot consulta **solo** la tabla socios. No responde colocaciones, transacciones, ni otros módulos.")
        campos = b_backend.get_campos_socios()
        render_campos_paginado_15(campos, ss_key="campos_socios")

    with col2:
        st.subheader("Filtros rápidos")

        regiones_all = b_backend.get_regiones()
        sucursales_all = b_backend.get_sucursales()
        _ = b_backend.get_sucursal_region_map()

        if "sel_region" not in st.session_state:
            st.session_state.sel_region = "—"
        if "sel_sucursal" not in st.session_state:
            st.session_state.sel_sucursal = "—"

        if st.session_state.sel_region != "—":
            sucursales_opts = ["—"] + b_backend.get_sucursales_por_region(st.session_state.sel_region)
        else:
            sucursales_opts = ["—"] + sucursales_all

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

        if st.session_state.sel_sucursal != "—" and st.session_state.sel_region != "—":
            if not b_backend.pertenece_sucursal_a_region(st.session_state.sel_sucursal, st.session_state.sel_region):
                region_correcta = b_backend.get_region_de_sucursal(st.session_state.sel_sucursal)
                if region_correcta:
                    st.info(f"🛠️ Ajusté **REGION** a **{region_correcta}** porque la sucursal seleccionada pertenece ahí.")
                    st.session_state.sel_region = region_correcta

        if st.session_state.sel_region != "—" and st.session_state.sel_sucursal != "—":
            if not b_backend.pertenece_sucursal_a_region(st.session_state.sel_sucursal, st.session_state.sel_region):
                st.warning("La sucursal seleccionada no pertenece a esa región. Se limpiará para evitar resultados vacíos.")
                st.session_state.sel_sucursal = "—"

        sel_region = st.session_state.sel_region
        sel_sucursal = st.session_state.sel_sucursal
        auto_inyectar = st.checkbox("Agregar estos filtros a mi pregunta", value=False)

# Ejemplos de uso
st.markdown('<div class="full-expander">', unsafe_allow_html=True)
with st.expander("💡 Ver ejemplos listos", expanded=False):
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
st.markdown('</div>', unsafe_allow_html=True)

# Conversación
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "rewriter_llm" not in st.session_state:
    st.session_state.rewriter_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

REWRITE_PROMPT = PromptTemplate.from_template(
    """Eres un reescritor de preguntas para análisis SOLO de la tabla socios.
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

# Entrada de usuario
prompt = st.chat_input("¿En qué te puedo ayudar?")
if prompt:
    user_prompt = prompt
    if auto_inyectar:
        extras = []
        if sel_region != "—": extras.append(f"en la región {sel_region}")
        if sel_sucursal != "—": extras.append(f"en la sucursal {sel_sucursal}")
        if extras: user_prompt = f"{user_prompt} ({', '.join(extras)})"

    st.session_state.mensajes.append({"role": "user", "content": user_prompt, "df": None})
    with st.chat_message("user"):
        st.write(user_prompt)

    block_msg = b_backend.detectar_bloqueo_texto_usuario(user_prompt)
    if block_msg:
        with st.chat_message("assistant"):
            st.error(block_msg)
        st.session_state.mensajes.append({"role": "assistant", "content": block_msg, "df": None})
    else:
        if b_backend.es_pregunta_contexto(user_prompt):
            respuesta_ctx = b_backend.responder_contexto_desde_txt(user_prompt)
            with st.chat_message("assistant"):
                st.write(respuesta_ctx)
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta_ctx, "df": None})
        else:
            pregunta_final = reescribir_pregunta_si_aplica(user_prompt)
            if pregunta_final == "FUERA_DE_ALCANCE":
                with st.chat_message("assistant"):
                    st.warning("Fuera de alcance: este bot consulta solo la tabla socios. Revisa los campos disponibles arriba.")
                st.session_state.mensajes.append({"role": "assistant", "content": "Fuera de alcance. Usa los campos de socios.", "df": None})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        texto, df, sql = b_backend.consulta(pregunta_final)

                    if (texto or "").startswith("🚫 Acción bloqueada por seguridad") or \
                       "bloqueada por seguridad" in (texto or "").lower():
                        st.error(texto)
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

                st.session_state.mensajes.append({"role": "assistant", "content": texto, "df": df})

# Reinicio
if st.button("🧹 Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()
