# Detecção de Estrelas em Imagens de Constelações (NOIRLab)

Projeto da disciplina **Processamento de Imagens e Sinais + IA Generativa** (Prof. Vinicius Santos Andrade e Patrick Pedreira Silva).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaTiosso/processamento-imagens-lab/blob/main/processamento_imagens.ipynb)

## Integrantes
* Fernando Fleuri Barbosa
* Isabela Xavier Tiosso
* Lucas Pomini Galli 

## Problema
Fotos de constelações têm milhares de estrelas fracas, Via Láctea, nebulosas, gradiente de fundo e ruído de sensor.
Objetivo: **isolar as estrelas** e **destacar as estrelas principais** que desenham a constelação.

## Pipeline

```
imagem RGB ─► escala de cinza ─► FFT 2D ─► filtro passa-faixa gaussiano ─► IFFT
                                          (tira fundo = baixa freq.
                                           e ruído   = alta freq.)
   ─► limiar k·σ robusto ─► abertura morfológica ─► componentes conexos
   ─► centroide, área, fluxo de cada estrela ─► ranking das estrelas principais
```

| Etapa | Técnica | Parâmetros |
|---|---|---|
| Pré-processamento | RGB → cinza (luminância) | — |
| **FFT** | passa-faixa gaussiano, com espelhamento de bordas | d0 = N/100, d1 = N/6 (ciclos/imagem) |
| **Segmentação** | limiar mediana + k·1,4826·MAD; abertura 2×2; componentes 8-conexos | k = 5, área mín. = 3 px |
| Decisão | ordenação por fluxo | top-12 |

## Resultados (céu sintético com 60 estrelas conhecidas, F1)

| Cenário | Gauss + Otsu (original) | FFT + Otsu | **FFT + k·σ** |
|---|---|---|---|
| limpo | 1,00 | 0,99 | **1,00** |
| gradiente médio | 0,21 | 0,99 | **1,00** |
| gradiente forte | 0,17 | 0,03 | **0,99** |

Na foto real de Órion, as estrelas de maior fluxo incluem Rigel, Betelgeuse e Bellatrix. Sem a FFT, a “estrela” mais brilhante é a própria Via Láctea (uma região de 412 mil px).
As figuras ficam em [`resultados/`](resultados/).

## Estrutura
- `processamento_imagens.ipynb` — notebook principal (roda no Colab ou localmente)
- `src/pipeline.py` — funções do pipeline
- `app.py` — aplicativo web (Streamlit) com o pipeline + análise por IA generativa (Gemini)
- `src/ia.py` — integração com o Gemini (prompt e chamada da API)
- `tests/` — testes (`python -m pytest`)
- `dados/` — imagens NOIRLab usadas (Órion, Escorpião)
- `legado/` — notebook exploratório inicial (Otsu, K-means, Watershed, nitidez)
- `construir_notebook.py` — regenera e executa o notebook
- `apresentacao/` — slides e roteiro de fala

## Dataset
NOIRLab — Constellations: https://noirlab.edu/public/education/constellations

## Como rodar
```bash
pip install -r requirements.txt
python -m pytest
python construir_notebook.py
```

## Aplicativo com IA generativa (Streamlit + Gemini)
O app junta o nosso pipeline de processamento de imagens com uma IA generativa:

1. O usuário escolhe uma foto de exemplo (Órion, Escorpião) ou envia a sua.
2. O pipeline (FFT passa-faixa + limiar k·σ) detecta as estrelas e marca as principais.
3. O **Gemini** recebe a foto marcada e a lista de posições/brilhos, e diz qual constelação é mais provável, com nível de confiança.
4. Opcionalmente, o app compara com a IA vendo **só a foto original** (sem o pipeline), para mostrar o efeito do processamento.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="sua-chave"   # gere em aistudio.google.com (nunca coloque a chave no código)
streamlit run app.py
```
A chave também pode ser colada na barra lateral do app, ou guardada em `.streamlit/secrets.toml` (modelo em `.streamlit/secrets.toml.example`; esse arquivo está no `.gitignore`).
O app usa `gemini-3.6-flash` por padrão e tenta modelos alternativos automaticamente em erros temporários ou quando o modelo não está disponível para a chave. Para definir o modelo principal, use `export GEMINI_MODEL="nome-do-modelo"` antes de iniciar o app.
A IA pode errar: confira sempre com uma carta celeste.
