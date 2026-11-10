"""Integração com IA generativa (Gemini) para interpretar o resultado do pipeline.

A IA recebe a foto com as estrelas principais marcadas (saída do nosso pipeline FFT)
e a lista de posições/brilhos, e responde qual constelação é mais provável.
Também há um prompt "sem pipeline" (só a foto original) para comparar.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from PIL import Image

MODELO_PADRAO = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MODELOS_FALLBACK = ("gemini-3.5-flash-lite", "gemini-3.8-flash")
CODIGOS_RETENTAVEIS = {404, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ResultadoGemini:
    texto: str
    modelo: str


PROMPT_SEM_PIPELINE = (
    "Você é um assistente de astronomia. A imagem é uma fotografia de campo largo do céu noturno. "
    "Diga qual constelação aparece com mais probabilidade e quais estrelas brilhantes você consegue "
    "reconhecer. Responda em português, para um estudante, em no máximo 10 linhas. "
    "Informe seu nível de confiança (alto, médio ou baixo) e, se não tiver certeza, diga isso claramente."
)


def montar_prompt(componentes: list[dict], largura: int, altura: int, n: int = 12,
                  pergunta_extra: str = "") -> str:
    """Texto enviado ao Gemini junto com a imagem marcada.

    `componentes` vem de `pipeline(...)['componentes']`, já ordenado por fluxo (brilho).
    O brilho é normalizado: 1.00 = a estrela mais brilhante detectada.
    """
    top = componentes[:n]
    if not top:
        raise ValueError("Nenhuma estrela detectada para enviar à IA.")
    f0 = top[0]["fluxo"] or 1.0
    linhas = [
        f"{i}. posição (x={e['x']:.0f}, y={e['y']:.0f}), brilho relativo {e['fluxo'] / f0:.2f}"
        for i, e in enumerate(top, 1)
    ]
    texto = (
        "Você é um assistente de astronomia. A imagem é uma fotografia de campo largo do céu noturno. "
        f"Um pipeline de processamento de imagens (FFT passa-faixa + limiar k·σ) detectou as {len(top)} estrelas "
        "mais brilhantes e as marcou com círculos numerados de 1 (mais brilhante) a "
        f"{len(top)} na imagem. Posições em pixels (origem no canto superior esquerdo; imagem {largura}x{altura}):\n"
        + "\n".join(linhas)
        + "\n\nCom base no padrão geométrico dessas estrelas e no brilho relativo, responda em português, "
        "para um estudante, em no máximo 10 linhas:\n"
        "1) Qual constelação é mais provável?\n"
        "2) Quais estrelas numeradas você associa a estrelas conhecidas? Só cite se tiver boa confiança.\n"
        "3) Nível de confiança (alto, médio ou baixo) e limitações da análise.\n"
        "Se não tiver certeza, diga isso claramente em vez de inventar."
    )
    if pergunta_extra.strip():
        texto += f"\n\nPergunta adicional do usuário: {pergunta_extra.strip()}"
    return texto


def analisar_com_gemini(imagem: Image.Image, prompt: str, api_key: str,
                        modelo: str = MODELO_PADRAO) -> ResultadoGemini:
    """Envia imagem + texto ao Gemini, tentando modelos alternativos em falhas recuperáveis."""

    if not api_key:
        raise ValueError("Falta a chave da API do Gemini.")
    from google import genai

    cliente = genai.Client(api_key=api_key)
    modelos = tuple(dict.fromkeys((modelo, *MODELOS_FALLBACK)))
    ultimo_erro = None

    for indice, modelo_atual in enumerate(modelos):
        try:
            resposta = cliente.models.generate_content(
                model=modelo_atual, contents=[imagem, prompt]
            )
        except Exception as erro:
            codigo = getattr(erro, "code", None)
            if codigo is None:
                codigo = getattr(getattr(erro, "response", None), "status_code", None)
            try:
                codigo = int(codigo)
            except (TypeError, ValueError):
                raise
            if codigo not in CODIGOS_RETENTAVEIS:
                raise
            ultimo_erro = erro
            if indice < len(modelos) - 1:
                time.sleep(min(2 ** indice, 4))
                continue
            break

        texto = (resposta.text or "").strip()
        if not texto:
            raise RuntimeError("A IA não devolveu texto (resposta vazia ou bloqueada).")
        return ResultadoGemini(texto=texto, modelo=modelo_atual)

    raise RuntimeError(
        f"Gemini indisponível nos modelos tentados: {', '.join(modelos)}. "
        f"Último erro: {ultimo_erro}"
    ) from ultimo_erro
