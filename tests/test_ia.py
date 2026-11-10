import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from src.ia import (
    MODELO_PADRAO,
    PROMPT_SEM_PIPELINE,
    analisar_com_gemini,
    montar_prompt,
)


def _comps(n):
    # componentes já ordenados por fluxo, como sai de pipeline(...)['componentes']
    return [{"x": 10.0 * i, "y": 5.0 * i, "area": 9, "fluxo": 1000.0 / i} for i in range(1, n + 1)]


def test_prompt_lista_so_as_n_principais_e_normaliza_o_brilho():
    prompt = montar_prompt(_comps(30), 1280, 1280, n=12)
    assert "1. posição" in prompt and "12. posição" in prompt
    assert "13. posição" not in prompt
    assert "brilho relativo 1.00" in prompt  # a mais brilhante vale 1.00
    assert "1280x1280" in prompt


def test_prompt_inclui_pergunta_extra_so_quando_existe():
    com = montar_prompt(_comps(5), 100, 100, n=5, pergunta_extra="como achar no céu?")
    sem = montar_prompt(_comps(5), 100, 100, n=5)
    assert "como achar no céu?" in com
    assert "Pergunta adicional" not in sem


def test_prompt_sem_estrelas_gera_erro_claro():
    with pytest.raises(ValueError):
        montar_prompt([], 100, 100)


def test_chamada_sem_chave_gera_erro_antes_de_ir_para_a_rede():
    with pytest.raises(ValueError):
        analisar_com_gemini(None, "oi", api_key="")


def test_falha_503_tenta_modelo_alternativo(monkeypatch):
    chamadas = []

    class ErroAPI(Exception):
        code = 503

    class Modelos:
        def generate_content(self, *, model, contents):
            chamadas.append(model)
            if model == MODELO_PADRAO:
                raise ErroAPI("sobrecarga")
            return SimpleNamespace(text="Resposta")

    genai_falso = SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(models=Modelos()))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai_falso))
    monkeypatch.setattr("src.ia.time.sleep", lambda _: None)

    resultado = analisar_com_gemini(Image.new("RGB", (1, 1)), "prompt", "chave")

    assert resultado.texto == "Resposta"
    assert resultado.modelo == "gemini-3.5-flash-lite"
    assert chamadas == [MODELO_PADRAO, "gemini-3.5-flash-lite"]


def test_erro_nao_recuperavel_nao_tenta_outro_modelo(monkeypatch):
    class ErroAPI(Exception):
        code = 400

    erro_esperado = ErroAPI("requisição inválida")

    class Modelos:
        def generate_content(self, *, model, contents):
            raise erro_esperado

    genai_falso = SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(models=Modelos()))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai_falso))

    with pytest.raises(ErroAPI) as erro:
        analisar_com_gemini(Image.new("RGB", (1, 1)), "prompt", "chave")

    assert erro.value is erro_esperado


def test_prompt_sem_pipeline_pede_confianca():
    assert "confiança" in PROMPT_SEM_PIPELINE
