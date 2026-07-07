"""
Funções auxiliares do projeto (multi-provider OpenAI/Gemini).

Centraliza:
- Carregamento de variáveis de ambiente (.env)
- Criação de LLMs (responder e juiz) provider-agnóstica via ``init_chat_model``
- Leitura/escrita de prompts em YAML <-> ChatPromptTemplate
- Leitura de datasets JSONL
- Execução de prompts com contrato uniforme ``{"output": str}``
- Parsing defensivo de JSON vindo do LLM

Baseado no padrão de ``samples/.../7-evaluation/shared/`` (clients, prompts,
parsers), estendido para suportar Gemini além de OpenAI.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Carrega o .env uma única vez, na importação.
load_dotenv()

# ------------------------------------------------------------------------- #
# Caminhos do projeto
# ------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
DATASETS_DIR = ROOT_DIR / "datasets"

PROMPT_V1_PATH = PROMPTS_DIR / "bug_to_user_story_v1.yml"
PROMPT_V2_PATH = PROMPTS_DIR / "bug_to_user_story_v2.yml"
DATASET_PATH = DATASETS_DIR / "bug_to_user_story.jsonl"

DATASET_NAME = "bug_to_user_story"


# ------------------------------------------------------------------------- #
# Configuração via ambiente (com defaults sensatos = Gemini free)
# ------------------------------------------------------------------------- #
def get_provider() -> str:
    """Provedor do LLM que RESPONDE (gera a user story)."""
    return os.getenv("LLM_PROVIDER", "google_genai")


def get_model_name() -> str:
    """Modelo que responde (default: gemini-2.5-flash)."""
    return os.getenv("LLM_MODEL", "gemini-2.5-flash")


def get_temperature() -> float:
    return float(os.getenv("LLM_TEMPERATURE", "0"))


def get_judge_provider() -> str:
    """Provedor do LLM que AVALIA (LLM-as-judge). Default: mesmo do responder."""
    return os.getenv("JUDGE_PROVIDER", get_provider())


def get_judge_model() -> str:
    """Modelo juiz (default: gemini-2.5-flash)."""
    return os.getenv("JUDGE_MODEL", "gemini-2.5-flash")


def get_username() -> Optional[str]:
    """Handle do LangSmith definido no ambiente (``LANGSMITH_USERNAME``)."""
    return os.getenv("LANGSMITH_USERNAME") or os.getenv("LANGCHAIN_USERNAME")


# Cache da descoberta do handle (evita repetir a chamada /settings).
_TENANT_HANDLE = None
_TENANT_HANDLE_RESOLVED = False


def get_tenant_handle(client=None) -> Optional[str]:
    """
    Descobre o handle público (``tenant_handle``) da conta via API do LangSmith.

    Usa ``client._get_settings()`` (mesma fonte que o ``push_prompt`` consulta
    internamente). Retorna ``None`` se não houver handle configurado ou se a
    chamada falhar (ex.: sem credenciais/rede).
    """
    global _TENANT_HANDLE, _TENANT_HANDLE_RESOLVED
    if _TENANT_HANDLE_RESOLVED:
        return _TENANT_HANDLE
    try:
        client = client or get_langsmith_client()
        settings = client._get_settings()
        _TENANT_HANDLE = getattr(settings, "tenant_handle", None) or None
    except Exception:  # noqa: BLE001 — descoberta é best-effort
        _TENANT_HANDLE = None
    _TENANT_HANDLE_RESOLVED = True
    return _TENANT_HANDLE


def resolve_username(client=None) -> Optional[str]:
    """
    Resolve o handle a usar no prompt identifier: primeiro ``LANGSMITH_USERNAME``
    do ambiente; se vazio, descobre automaticamente via API. ``None`` se nenhum.
    """
    return get_username() or get_tenant_handle(client)


# ------------------------------------------------------------------------- #
# Fábricas de clientes / LLMs (provider-agnósticas)
# ------------------------------------------------------------------------- #
def make_llm(model: Optional[str] = None, provider: Optional[str] = None,
             temperature: Optional[float] = None):
    """
    Cria um chat model do LangChain de forma provider-agnóstica.

    Usa ``init_chat_model`` para que o mesmo código sirva a OpenAI, Gemini e
    Groq, mantendo o tracing do LangSmith via callbacks (LANGCHAIN_TRACING_V2).

    OpenAI:  provider="openai"        model="gpt-4o-mini" | "gpt-4o"
    Gemini:  provider="google_genai"  model="gemini-2.5-flash"
    Groq:    provider="groq"          model="qwen/qwen3-32b"
    """
    from langchain.chat_models import init_chat_model

    provider = provider or get_provider()
    model = model or get_model_name()
    temperature = get_temperature() if temperature is None else temperature

    extra = {}
    if provider == "groq":
        # Modelos de reasoning (Qwen3) emitem blocos <think>; "hidden" faz o
        # Groq remover o raciocínio e devolver só a resposta final.
        extra["reasoning_format"] = os.getenv("GROQ_REASONING_FORMAT", "hidden")
        # Free tier tem TPM baixo (6000 tok/min); retries generosos respeitam o
        # backoff "try again in ~22s" para a avaliação completar mesmo assim.
        extra["max_retries"] = int(os.getenv("GROQ_MAX_RETRIES", "12"))

    return init_chat_model(model, model_provider=provider,
                           temperature=temperature, **extra)


def make_judge_llm():
    """LLM juiz (temperatura 0 para consistência nas notas)."""
    return make_llm(model=get_judge_model(), provider=get_judge_provider(),
                    temperature=0)


def get_langsmith_client():
    """Cliente do LangSmith (lê LANGCHAIN_API_KEY do ambiente)."""
    from langsmith import Client

    return Client()


# ------------------------------------------------------------------------- #
# Prompts em YAML  <->  ChatPromptTemplate
# ------------------------------------------------------------------------- #
def load_prompt_config(path) -> dict:
    """Lê o YAML de um prompt e devolve o dict cru."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_chat_template(config: dict):
    """
    Converte o dict do YAML (chave ``messages: [{role, content}]``) em um
    ``ChatPromptTemplate``. Mesmo formato aceito por ``client.push_prompt``.
    """
    from langchain_core.prompts import ChatPromptTemplate

    if "messages" not in config:
        raise ValueError("YAML de prompt não contém a chave 'messages'.")
    messages = [(m["role"], m["content"]) for m in config["messages"]]
    return ChatPromptTemplate.from_messages(messages)


def load_chat_prompt(path):
    """Atalho: carrega o YAML e devolve o ChatPromptTemplate."""
    return config_to_chat_template(load_prompt_config(path))


def chat_template_to_config(prompt_obj, input_variables=None) -> dict:
    """
    Serializa um ``ChatPromptTemplate`` (ex.: vindo de ``client.pull_prompt``)
    de volta para o dict YAML no formato ``messages: [{role, content}]``.
    """
    messages = []
    for m in prompt_obj.messages:
        cls = m.__class__.__name__
        if "System" in cls:
            role = "system"
        elif "AI" in cls or "Assistant" in cls:
            role = "assistant"
        else:
            role = "user"
        template = getattr(getattr(m, "prompt", None), "template", None)
        if template is None:
            template = getattr(m, "content", "")
        messages.append({"role": role, "content": template})

    if input_variables is None:
        input_variables = list(getattr(prompt_obj, "input_variables", []) or [])

    return {
        "_type": "prompt",
        "input_variables": list(input_variables),
        "template_format": "f-string",
        "messages": messages,
    }


def save_prompt_config(config: dict, path) -> None:
    """Grava o dict de prompt em YAML (unicode preservado, ordem estável)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=100)


# ------------------------------------------------------------------------- #
# Dataset JSONL
# ------------------------------------------------------------------------- #
def load_jsonl(path) -> list:
    """Lê um arquivo JSONL e devolve a lista de objetos (uma linha = um objeto)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Chaves aceitas para o texto do bug no dataset (o prompt v1 do Hub usa
# "bug_report"; mantemos "bug" como alias por robustez).
BUG_INPUT_KEYS = ("bug_report", "bug")


def extract_bug(inputs: dict) -> str:
    """Extrai o texto do bug dos inputs, tolerante à chave (bug_report/bug)."""
    inputs = inputs or {}
    for key in BUG_INPUT_KEYS:
        if inputs.get(key):
            return inputs[key]
    # fallback: primeiro valor string não-vazio
    for value in inputs.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def upload_langsmith_dataset(dataset_file, dataset_name: str, description: str,
                             client=None) -> int:
    """
    Sobe (idempotentemente) um dataset JSONL para o LangSmith, com metadata.

    Se o dataset já existir, remove os exemplos antigos e recria; caso contrário
    cria o dataset. Baseado em ``samples/.../7-evaluation/shared/datasets.py``.
    Retorna o número de exemplos enviados.
    """
    client = client or get_langsmith_client()
    examples = load_jsonl(dataset_file)

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        for ex in client.list_examples(dataset_name=dataset_name):
            client.delete_example(ex.id)
    except Exception:  # noqa: BLE001 — dataset ainda não existe
        dataset = client.create_dataset(dataset_name=dataset_name,
                                        description=description)

    for ex in examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata=ex.get("metadata", {}),
            dataset_id=dataset.id,
        )
    return len(examples)


# ------------------------------------------------------------------------- #
# Execução do prompt — contrato uniforme {"output": str}
# ------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """
    Remove blocos de raciocínio ``<think>...</think>`` (modelos de reasoning,
    ex.: Qwen3 via Groq). Defesa extra além do ``reasoning_format='hidden'``.
    """
    text = _THINK_RE.sub("", text or "")
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def run_prompt(prompt_obj, llm, **format_kwargs) -> dict:
    """
    Formata o ChatPromptTemplate com ``format_kwargs`` (ex.: ``bug_report=...``),
    invoca o LLM e devolve ``{"output": <conteúdo>}`` (sem blocos <think>).

    Esse é o contrato lido por todos os evaluators (``run.outputs["output"]``).
    """
    messages = prompt_obj.format_messages(**format_kwargs)
    response = llm.invoke(messages)
    content = getattr(response, "content", response)
    content = content if isinstance(content, str) else str(content)
    return {"output": strip_reasoning(content)}


# ------------------------------------------------------------------------- #
# Parsing defensivo de JSON (de shared/parsers.py)
# ------------------------------------------------------------------------- #
def parse_json_response(text: str) -> dict:
    """
    Extrai JSON de uma resposta de LLM, removendo cercas markdown se houver.
    Nunca levanta exceção: devolve ``{}`` quando não consegue parsear.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
