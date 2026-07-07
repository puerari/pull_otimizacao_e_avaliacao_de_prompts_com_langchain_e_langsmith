"""
Testes de validação do prompt otimizado (prompts/bug_to_user_story_v2.yml).

Executa OFFLINE (não chama LLM nem LangSmith): valida a estrutura e o conteúdo
do YAML do prompt v2, conforme exigido pelo desafio.

    pytest tests/test_prompts.py
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPT_V2_PATH = ROOT / "prompts" / "bug_to_user_story_v2.yml"


# ------------------------------------------------------------------------- #
# Fixtures
# ------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config() -> dict:
    """Carrega o YAML do prompt v2."""
    assert PROMPT_V2_PATH.exists(), f"Prompt não encontrado: {PROMPT_V2_PATH}"
    with open(PROMPT_V2_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def messages_by_role(config) -> dict:
    """Mapa role -> content das mensagens do prompt."""
    return {m["role"]: m["content"] for m in config.get("messages", [])}


@pytest.fixture(scope="module")
def full_text(config) -> str:
    """Concatenação de todo o texto das mensagens (para buscas de conteúdo)."""
    return "\n".join(m.get("content", "") for m in config.get("messages", []))


# ------------------------------------------------------------------------- #
# Os 6 testes obrigatórios
# ------------------------------------------------------------------------- #
def test_prompt_has_system_prompt(messages_by_role):
    """Verifica se o campo system existe e não está vazio."""
    assert "system" in messages_by_role, "O prompt deve conter uma mensagem 'system'."
    assert messages_by_role["system"].strip() != "", "O system prompt está vazio."


def test_prompt_has_role_definition(full_text):
    """Verifica se o prompt define uma persona (ex.: 'Você é uma Product Owner')."""
    assert re.search(r"voc[êe]\s+[ée]\s+um[a]?\b", full_text, re.IGNORECASE), (
        "O prompt deve definir uma persona explícita (ex.: 'Você é uma Product Owner')."
    )


def test_prompt_mentions_format(full_text):
    """Verifica se o prompt exige formato Markdown ou User Story padrão."""
    low = full_text.lower()
    assert ("markdown" in low) or ("user story" in low), (
        "O prompt deve exigir formato Markdown ou User Story padrão."
    )


def test_prompt_has_few_shot_examples(full_text):
    """Verifica se o prompt contém exemplos de entrada/saída (Few-shot)."""
    low = full_text.lower()
    n_exemplos = low.count("exemplo")
    n_bugs = len(re.findall(r"bug\s*:", low))
    assert n_exemplos >= 2 or n_bugs >= 2, (
        "O prompt deve conter pelo menos 2 exemplos de entrada/saída (Few-shot)."
    )
    # Deve haver o par entrada (Bug) -> saída (User Story) nos exemplos.
    assert "user story" in low and "bug" in low, (
        "Os exemplos few-shot devem mostrar o par Bug -> User Story."
    )


def test_prompt_no_todos(full_text):
    """Garante que não sobrou nenhum marcador [TODO]/FIXME no texto."""
    assert "[TODO]" not in full_text and "[todo]" not in full_text, (
        "Há um [TODO] pendente no prompt."
    )
    assert not re.search(r"\bTODO\b", full_text), "Há um marcador TODO pendente."
    assert not re.search(r"\bFIXME\b", full_text), "Há um marcador FIXME pendente."


def test_minimum_techniques(config):
    """Verifica (via metadados do YAML) se pelo menos 2 técnicas foram listadas."""
    techniques = config.get("metadata", {}).get("techniques", [])
    assert isinstance(techniques, list), "metadata.techniques deve ser uma lista."
    assert len(techniques) >= 2, (
        f"Devem ser listadas ao menos 2 técnicas; encontrado: {techniques}"
    )
