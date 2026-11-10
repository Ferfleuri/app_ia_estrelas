"""Aplicativo web: detecção de estrelas (FFT passa-faixa + limiar k·σ) + análise com IA generativa (Gemini).

Como rodar:  streamlit run app.py
A chave do Gemini NUNCA vai no código: use a variável de ambiente GEMINI_API_KEY,
o arquivo .streamlit/secrets.toml (local) ou os Secrets do Streamlit Cloud.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.ia import MODELO_PADRAO, PROMPT_SEM_PIPELINE, analisar_com_gemini, montar_prompt
from src.pipeline import espectro_magnitude, pipeline

st.set_page_config(page_title="Detector de Estrelas + IA", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #0f2342 0%, #071421 30%, #030b14 100%);
            color: #edf6ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(10, 22, 36, 0.88);
            border-right: 1px solid rgba(117, 168, 255, 0.2);
        }
        .stHeader, .stSubheader {
            color: #f2f7ff;
        }
        .stMetric {
            background: rgba(15, 30, 48, 0.75);
            border: 1px solid rgba(124, 170, 255, 0.18);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 10px 24px rgba(12, 19, 32, 0.22);
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #dfeeff;
        }
        .stExpander {
            background: rgba(11, 24, 38, 0.6);
            border: 1px solid rgba(122, 161, 255, 0.18);
            border-radius: 14px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #5ba7ff 0%, #7e7bff 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
            box-shadow: 0 10px 22px rgba(91, 167, 255, 0.25);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #7ab8ff 0%, #9092ff 100%);
            filter: brightness(1.02);
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(13, 24, 36, 0.8);
            border: 1px dashed rgba(115, 160, 255, 0.45);
            border-radius: 14px;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stSlider > div {
            background: rgba(10, 19, 31, 0.75);
            color: #edf6ff;
        }
        .stCaption {
            color: #b7c9dd;
        }
        .main .block-container {
            max-width: 1400px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RAIZ = Path(__file__).parent
EXEMPLOS = {
    "Órion (exemplo)": ("dados/orion.jpg", "Órion"),
    "Escorpião (exemplo)": ("dados/scorpius.jpg", "Escorpião"),
}
EXEMPLOS_DISPONIVEIS = {
    nome: (rel, label)
    for nome, (rel, label) in EXEMPLOS.items()
    if (RAIZ / rel).exists()
}
LADO_MAX = 1280  # imagens maiores são reduzidas (a FFT fica mais rápida)


# ------------------------------------------------------------------ utilidades
def chave_api() -> str:
    """Procura a chave nos Secrets do Streamlit e depois na variável de ambiente."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def carregar_rgb(origem) -> np.ndarray:
    img = Image.open(origem).convert("RGB")
    if max(img.size) > LADO_MAX:
        img.thumbnail((LADO_MAX, LADO_MAX))
    return np.array(img)


@st.cache_data(show_spinner="Rodando o pipeline (FFT + segmentação)...")
def rodar_pipeline(rgb: np.ndarray, k: float) -> dict:
    cinza = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return pipeline(cinza, k=k)


def marcar(rgb: np.ndarray, comps: list[dict], n: int) -> np.ndarray:
    """Desenha círculos numerados nas n estrelas mais brilhantes."""
    out = rgb.copy()
    h, w = out.shape[:2]
    raio = max(10, int(min(h, w) * 0.017))
    escala = max(0.5, min(h, w) / 1280)
    for i, e in enumerate(comps[:n], 1):
        c = (int(e["x"]), int(e["y"]))
        cv2.circle(out, c, raio, (0, 255, 120), max(2, int(3 * escala)))
        cv2.putText(out, str(i), (c[0] + raio + 2, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, (255, 255, 0), max(1, int(2 * escala)))
    return out


def para_exibir(img: np.ndarray) -> np.ndarray:
    """Normaliza uma imagem de ponto flutuante para 0-255 (só para mostrar na tela)."""
    x = np.clip(img, 0, None)
    topo = np.percentile(x, 99.5) or 1.0
    return np.uint8(np.clip(x / topo, 0, 1) * 255)


def reduzir(rgb: np.ndarray, lado: int = 1024) -> Image.Image:
    img = Image.fromarray(rgb)
    img.thumbnail((lado, lado))
    return img


# ------------------------------------------------------------------ barra lateral
st.sidebar.header("Configurações")
if not EXEMPLOS_DISPONIVEIS:
    st.sidebar.warning("Nenhuma imagem de exemplo foi encontrada em dados/. Envie sua própria foto do céu.")
opcoes = list(EXEMPLOS_DISPONIVEIS) + ["Enviar minha foto"]
fonte = st.sidebar.radio("Imagem", opcoes)
arquivo = None
if fonte == "Enviar minha foto":
    arquivo = st.sidebar.file_uploader("Foto do céu (JPG ou PNG)", type=["jpg", "jpeg", "png"])

st.sidebar.subheader("Parâmetros do pipeline")
k = st.sidebar.slider("k do limiar (mediana + k·σ)", 2.0, 14.0, 5.0, 0.5,
                      help="k baixo: o ruído vira estrela. k alto: estrelas fracas somem. Padrão do projeto: 5.")
n = st.sidebar.slider("Estrelas principais", 5, 20, 12)

st.sidebar.subheader("IA generativa")
chave = chave_api()
if chave:
    st.sidebar.success("Chave do Gemini habilitada e pronta para uso.")
else:
    chave = st.sidebar.text_input("Chave da API do Gemini", type="password",
                                  help="Gere em aistudio.google.com. Ela não é salva em nenhum arquivo.")
    if chave:
        st.sidebar.success("Chave do Gemini habilitada e pronta para uso.")
    else:
        st.sidebar.caption("Status: aguardando chave da API.")
st.sidebar.caption(f"Modelo principal: {MODELO_PADRAO}")

# ------------------------------------------------------------------ imagem de entrada
st.markdown(
    """
    <div style='padding: 0.25rem 0 1.2rem 0;'>
        <h1 style='margin-bottom: 0.2rem; color: #f3f8ff; font-size: 2.5rem;'>✨ Detector de Estrelas + IA</h1>
        <div style='color: #b6cbe1; font-size: 1rem;'>
            Pipeline de Processamento de Imagens (FFT passa-faixa + segmentação) integrado a uma IA generativa (Gemini).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

gabarito = None
if fonte in EXEMPLOS_DISPONIVEIS:
    caminho, gabarito = EXEMPLOS_DISPONIVEIS[fonte]
    rgb = carregar_rgb(RAIZ / caminho)
elif arquivo is not None:
    rgb = carregar_rgb(arquivo)
else:
    st.info("Envie uma foto do céu na barra lateral, ou adicione as imagens de exemplo em dados/.")
    st.stop()

res = rodar_pipeline(rgb, k)
comps = res["componentes"]
marcada = marcar(rgb, comps, n)

# ------------------------------------------------------------------ 1. pipeline
st.markdown("### 1. Processamento de imagem")
c1, c2, c3 = st.columns(3)
c1.image(rgb, caption="Imagem original", width="stretch")
c2.image(para_exibir(res["filtrada"]), caption="Depois da FFT passa-faixa (só o tamanho de estrela)",
         width="stretch")
c3.image(marcada, caption=f"As {min(n, len(comps))} estrelas principais", width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Estrelas detectadas", len(comps))
m2.metric("Limiar (mediana + k·σ)", f"{res['limiar']:.2f}")
m3.metric("Cortes da FFT (d0 / d1)", f"{res['d0']:.0f} / {res['d1']:.0f}")

with st.expander("Ver mais etapas (espectro e segmentação)"):
    e1, e2 = st.columns(2)
    e1.image(para_exibir(espectro_magnitude(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))),
             caption="Espectro de frequências (FFT)", width="stretch")
    e2.image(res["binaria"], caption="Segmentação (limiar + morfologia)", width="stretch")

if not comps:
    st.warning("Nenhuma estrela detectada. Tente diminuir o k na barra lateral.")
    st.stop()

# ------------------------------------------------------------------ 2. IA
st.markdown("### 2. Análise com IA generativa (Gemini)")
st.write("A IA recebe a foto com as estrelas principais marcadas pelo nosso pipeline, mais a lista de posições e brilhos, "
         "e diz qual constelação é mais provável.")
pergunta = st.text_input("Pergunta extra para a IA (opcional)", placeholder="Ex.: explique como localizar essa constelação no céu")
comparar = st.checkbox("Comparar com a IA vendo só a foto original (sem o pipeline)", value=True)

altura, largura = rgb.shape[:2]
prompt = montar_prompt(comps, largura, altura, n, pergunta)
with st.expander("Ver exatamente o que é enviado para a IA"):
    st.image(marcada, caption="Imagem enviada (com pipeline)", width=360)
    st.code(prompt, language="text")

if st.button("Analisar com IA", type="primary"):
    if not chave:
        st.error("Cole a chave da API do Gemini na barra lateral para continuar.")
    else:
        try:
            with st.spinner("Consultando o Gemini..."):
                resp_com = analisar_com_gemini(reduzir(marcada), prompt, chave)
                resp_sem = analisar_com_gemini(reduzir(rgb), PROMPT_SEM_PIPELINE, chave) if comparar else None
            st.session_state["resultado_ia"] = (fonte, resp_com, resp_sem)
            st.success("Conexão com Gemini validada. Análise concluída com sucesso.")
        except Exception as erro:  # mostra o motivo sem expor a chave
            st.error(f"Não foi possível consultar o Gemini: {erro}")

salvo = st.session_state.get("resultado_ia")
if salvo and salvo[0] == fonte:
    _, resp_com, resp_sem = salvo
    texto_com = getattr(resp_com, "texto", resp_com)
    modelo_com = getattr(resp_com, "modelo", MODELO_PADRAO)
    if gabarito:
        st.info(f"Gabarito desta imagem: **{gabarito}**")
    if resp_sem:
        texto_sem = getattr(resp_sem, "texto", resp_sem)
        modelo_sem = getattr(resp_sem, "modelo", MODELO_PADRAO)
        a, b = st.columns(2)
        a.subheader("Com pipeline (FFT)")
        a.caption(f"Modelo: {modelo_com}")
        a.write(texto_com)
        b.subheader("Sem pipeline (só a foto)")
        b.caption(f"Modelo: {modelo_sem}")
        b.write(texto_sem)
    else:
        st.subheader("Resposta da IA")
        st.caption(f"Modelo: {modelo_com}")
        st.write(texto_com)
    st.caption("A IA pode errar ou inventar detalhes. Confira sempre com uma carta celeste.")
