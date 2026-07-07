"""
As 5 métricas de avaliação do desafio.

Divisão (como no CLI do enunciado):
  - Métricas Base .....: F1-Score, Clarity, Precision
  - Métricas Derivadas : Helpfulness, Correctness

Implementação:
  - Precision / F1-Score : cálculo textual set-based sobre as CATEGORIAS DE
    REQUISITO (vocabulário controlado) extraídas da saída vs. as esperadas no
    dataset. Segue o padrão de ``samples/.../7-evaluation/2-precision/metrics.py``
    (comparar apenas vocabulário controlado, nunca prosa livre).
  - Helpfulness / Correctness / Clarity : LLM-as-judge provider-agnóstico
    (usa o LLM juiz configurado — Gemini ou OpenAI — via src/utils.py), com
    rubrica dedicada e nota normalizada 0.0-1.0. Correctness usa a referência
    (``example.outputs["reference"]``); os demais não.

Funções puras (extract_*, calculate_precision_recall_f1, parse) são testáveis
offline, sem chamar nenhum LLM.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Set

import utils

# Limiar de aprovação do desafio: TODAS as métricas devem ser >= 0.8.
THRESHOLD = 0.8

# Vocabulário controlado de categorias de requisito (fonte única de verdade;
# deve casar com a lista do prompt prompts/bug_to_user_story_v2.yml).
REQUIREMENT_CATEGORIES: Set[str] = {
    "input_validation", "error_handling", "user_feedback", "security",
    "data_integrity", "performance", "accessibility", "logging",
    "compatibility", "state_management", "concurrency", "localization",
}

# Ordem de exibição no relatório (agrupada como no enunciado).
DERIVED_METRICS = ["helpfulness", "correctness"]
BASE_METRICS = ["f1_score", "clarity", "precision"]
REPORT_METRICS = DERIVED_METRICS + BASE_METRICS


# ========================================================================= #
# Precision / Recall / F1 — set-based sobre categorias (vocabulário controlado)
# ========================================================================= #
def normalize_category(token: str) -> str:
    """Normaliza um rótulo de categoria: minúsculas, espaços/hífens -> underscore."""
    return re.sub(r"[\s\-]+", "_", (token or "").strip().lower())


def extract_predicted_categories(output: Dict) -> Set[str]:
    """
    Extrai o conjunto de categorias de requisito da saída gerada.

    Estratégia:
      1. Localiza a seção "## Categorias de Requisito" e lê os rótulos dela.
      2. Fallback: varre o texto inteiro por rótulos do vocabulário controlado.
    Só retorna rótulos que pertencem ao vocabulário (evita ruído).
    """
    text = (output or {}).get("output", "") or ""
    found: Set[str] = set()

    # 1) Seção dedicada
    match = re.search(
        r"##\s*Categorias\s+de\s+Requisito\s*(.+?)(?:\n\s*##|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    region = match.group(1) if match else text
    for token in re.split(r"[,\n;]+", region):
        cat = normalize_category(token.lstrip("-*• ").strip())
        if cat in REQUIREMENT_CATEGORIES:
            found.add(cat)

    # 2) Fallback: procurar rótulos do vocabulário em todo o texto
    if not found:
        low = text.lower()
        for cat in REQUIREMENT_CATEGORIES:
            if cat in low:
                found.add(cat)

    return found


def extract_expected_categories(example) -> Set[str]:
    """Conjunto de categorias esperadas do exemplo do dataset."""
    outputs = _example_outputs(example)
    return {normalize_category(c) for c in outputs.get("expected_categories", [])}


def calculate_precision_recall_f1(
    outputs: List[Dict],
    examples: List,
    extract_predicted: Callable[[Dict], Set] = extract_predicted_categories,
    extract_expected: Callable = extract_expected_categories,
) -> List[Dict]:
    """
    Precision, Recall e F1 micro-average (soma global de TP/FP/FN).

    Retorna lista no formato de summary evaluator do LangSmith:
      [{"key": "precision"|"recall"|"f1_score", "score": float, "comment": str}]
    """
    tp = fp = fn = 0
    for output, example in zip(outputs, examples):
        predicted = extract_predicted(output)
        expected = extract_expected(example)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return [
        {"key": "precision", "score": precision, "comment": f"TP:{tp} FP:{fp}"},
        {"key": "recall", "score": recall, "comment": f"TP:{tp} FN:{fn}"},
        {"key": "f1_score", "score": f1, "comment": f"P:{precision:.2f} R:{recall:.2f}"},
    ]


def precision_recall_f1_summary(outputs: List[Dict], examples: List) -> List[Dict]:
    """Summary evaluator para o ``evaluate()`` do LangSmith (P/R/F1 agregados)."""
    return calculate_precision_recall_f1(outputs, examples)


# ========================================================================= #
# LLM-as-judge — Helpfulness / Correctness / Clarity (provider-agnóstico)
# ========================================================================= #
_JUDGE_LLM = None  # cache lazy (evita instanciar o LLM na importação do módulo)


def _get_judge():
    global _JUDGE_LLM
    if _JUDGE_LLM is None:
        _JUDGE_LLM = utils.make_judge_llm()
    return _JUDGE_LLM


# Rubricas (0 a 10) por métrica.
_RUBRICS: Dict[str, str] = {
    "helpfulness": (
        "Avalie o quão ÚTIL e ACIONÁVEL é a user story gerada para um time de "
        "desenvolvimento resolver o bug. Considere: o valor está claro? Os "
        "critérios de aceite são testáveis e suficientes? A story é completa e "
        "específica (sem vaguidade)? Uma story vaga, incompleta ou genérica recebe "
        "nota baixa; uma story clara, específica e pronta para desenvolvimento "
        "recebe nota alta."
    ),
    "correctness": (
        "Compare a user story GERADA com a REFERÊNCIA (gabarito). Avalie se ela "
        "captura corretamente o ATOR, o OBJETIVO, o BENEFÍCIO e os critérios "
        "essenciais implicados pelo bug. NÃO exija correspondência literal — "
        "avalie equivalência semântica. Contradições ou omissões de pontos "
        "essenciais reduzem a nota."
    ),
    "clarity": (
        "Avalie a CLAREZA e a ESTRUTURA da user story. Ela segue o padrão "
        "'Como <ator>, quero <objetivo>, para <benefício>'? Está organizada em "
        "seções (Título, User Story, Critérios de Aceite)? A linguagem é "
        "inequívoca e os critérios de aceite são bem formados (Dado/Quando/Então)? "
        "Texto confuso, ambíguo ou desestruturado recebe nota baixa."
    ),
}

_JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso e imparcial de user stories de engenharia de "
    "software. Você atribui uma nota INTEIRA de 0 a 10 conforme a rubrica e "
    "responde ESTRITAMENTE em JSON, sem texto adicional, no formato: "
    '{{"score": <inteiro 0-10>, "justification": "<curta justificativa>"}}'
)


def _build_judge_messages(metric_key, bug, prediction, reference=None):
    from langchain_core.messages import HumanMessage, SystemMessage

    rubric = _RUBRICS[metric_key]
    parts = [
        f"MÉTRICA: {metric_key}",
        f"RUBRICA: {rubric}",
        "",
        "BUG REPORT (entrada):",
        bug or "(vazio)",
        "",
        "USER STORY GERADA (a ser avaliada):",
        prediction or "(vazio)",
    ]
    if reference is not None:
        parts += ["", "REFERÊNCIA (gabarito):", reference or "(vazio)"]
    parts += [
        "",
        "Responda APENAS o JSON com 'score' (inteiro de 0 a 10) e 'justification'.",
    ]
    # NOTA: o system usa chaves duplas por ser exemplo de JSON; aqui é string comum.
    system = _JUDGE_SYSTEM.replace("{{", "{").replace("}}", "}")
    return [SystemMessage(content=system), HumanMessage(content="\n".join(parts))]


def parse_judge_score(text: str):
    """
    Extrai (score_normalizado_0a1, justificativa) da resposta do juiz.

    O juiz pontua de 0 a 10; normalizamos dividindo por 10 (equivalente ao
    ``normalize_by: 10`` dos evaluators do LangChain). Robusto a JSON com cerca
    markdown; fallback por regex para o primeiro número.
    """
    data = utils.parse_json_response(text)
    justification = (data.get("justification") or data.get("reason") or "").strip()
    raw = data.get("score", None)

    if raw is None:
        m = re.search(r"-?\d+(?:\.\d+)?", text or "")
        raw = m.group(0) if m else 0
        if not justification:
            justification = (text or "").strip()[:200]

    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0

    score = max(0.0, min(1.0, value / 10.0))
    return score, justification


def make_judge_evaluator(metric_key: str, use_reference: bool = False) -> Callable:
    """
    Cria um evaluator per-exemplo (assinatura ``(run, example)``) compatível com
    ``langsmith.evaluate()``, que usa o LLM juiz configurado.
    """
    def _evaluator(run, example) -> Dict:
        prediction = (getattr(run, "outputs", None) or {}).get("output", "")
        bug = utils.extract_bug(getattr(example, "inputs", None) or {})
        reference = None
        if use_reference:
            reference = _example_outputs(example).get("reference", "")
        try:
            messages = _build_judge_messages(metric_key, bug, prediction, reference)
            response = _get_judge().invoke(messages)
            text = utils.strip_reasoning(getattr(response, "content", str(response)))
        except Exception as exc:  # noqa: BLE001
            return {"key": metric_key, "score": 0.0, "comment": f"judge error: {exc}"}
        score, justification = parse_judge_score(text)
        return {"key": metric_key, "score": score, "comment": justification}

    _evaluator.__name__ = f"{metric_key}_judge"
    return _evaluator


# ========================================================================= #
# Juiz COMBINADO — avalia Helpfulness/Correctness/Clarity numa ÚNICA chamada
# (reduz ~3x o número de chamadas ao LLM; essencial sob rate limit apertado)
# ========================================================================= #
_COMBINED_JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso e imparcial de user stories de engenharia de "
    "software. Avalie a user story gerada em TRÊS métricas, cada uma com nota "
    "INTEIRA de 0 a 10, e responda ESTRITAMENTE em JSON, sem texto adicional, "
    'no formato: {{"helpfulness": {{"score": <0-10>, "justification": "..."}}, '
    '"correctness": {{"score": <0-10>, "justification": "..."}}, '
    '"clarity": {{"score": <0-10>, "justification": "..."}}}}'
)


def _build_combined_judge_messages(bug, prediction, reference):
    from langchain_core.messages import HumanMessage, SystemMessage

    parts = [
        "Avalie a USER STORY GERADA nas três métricas abaixo (0 a 10 cada).",
        "",
        f"[helpfulness] {_RUBRICS['helpfulness']}",
        f"[correctness] {_RUBRICS['correctness']}",
        f"[clarity] {_RUBRICS['clarity']}",
        "",
        "BUG REPORT (entrada):",
        bug or "(vazio)",
        "",
        "USER STORY GERADA (a avaliar):",
        prediction or "(vazio)",
        "",
        "REFERÊNCIA (gabarito — use para correctness):",
        reference or "(vazio)",
        "",
        "Responda APENAS o JSON com as três métricas.",
    ]
    system = _COMBINED_JUDGE_SYSTEM.replace("{{", "{").replace("}}", "}")
    return [SystemMessage(content=system), HumanMessage(content="\n".join(parts))]


def _score_from_entry(data: Dict, key: str):
    """Extrai (score 0-1, justificativa) de data[key] no JSON do juiz combinado."""
    entry = data.get(key)
    if isinstance(entry, dict):
        raw = entry.get("score")
        just = (entry.get("justification") or entry.get("reason") or "").strip()
    else:
        raw = entry
        just = ""
    try:
        score = max(0.0, min(1.0, float(raw) / 10.0))
    except (TypeError, ValueError):
        score = 0.0
    return score, just


def combined_judge_evaluator(run, example) -> Dict:
    """
    Evaluator multi-métrica: uma única chamada ao juiz devolve Helpfulness,
    Correctness e Clarity. Retorna ``{"results": [...]}`` (formato aceito pelo
    ``evaluate()`` do LangSmith para múltiplas métricas por evaluator).
    """
    prediction = (getattr(run, "outputs", None) or {}).get("output", "")
    bug = utils.extract_bug(getattr(example, "inputs", None) or {})
    reference = _example_outputs(example).get("reference", "")

    keys = ("helpfulness", "correctness", "clarity")
    try:
        messages = _build_combined_judge_messages(bug, prediction, reference)
        response = _get_judge().invoke(messages)
        text = utils.strip_reasoning(getattr(response, "content", str(response)))
    except Exception as exc:  # noqa: BLE001
        return {"results": [{"key": k, "score": 0.0, "comment": f"judge error: {exc}"}
                            for k in keys]}

    data = utils.parse_json_response(text)
    results = []
    for k in keys:
        score, just = _score_from_entry(data, k)
        results.append({"key": k, "score": score, "comment": just or "(sem justificativa)"})
    return {"results": results}


# ========================================================================= #
# Coleções prontas para o evaluate.py
# ========================================================================= #
def get_evaluators(combined: bool = True) -> List[Callable]:
    """
    Evaluators per-exemplo (LLM-as-judge): Helpfulness, Correctness, Clarity.

    combined=True (padrão): um único evaluator faz 1 chamada e retorna as 3
    métricas — mais econômico sob rate limit. combined=False: 3 evaluators
    separados (3 chamadas por exemplo).
    """
    if combined:
        return [combined_judge_evaluator]
    return [
        make_judge_evaluator("helpfulness", use_reference=False),
        make_judge_evaluator("correctness", use_reference=True),
        make_judge_evaluator("clarity", use_reference=False),
    ]


def get_summary_evaluators() -> List[Callable]:
    """Summary evaluators (set-based): precision, recall, f1_score."""
    return [precision_recall_f1_summary]


# ========================================================================= #
# Helpers internos
# ========================================================================= #
def _example_outputs(example) -> Dict:
    """
    Normaliza o acesso a outputs tanto para objetos do LangSmith (``example.outputs``)
    quanto para dicts crus do JSONL (``example["outputs"]``).
    """
    if isinstance(example, dict):
        return example.get("outputs", {}) or {}
    return getattr(example, "outputs", {}) or {}
