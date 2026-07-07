"""
Avaliação automática dos prompts no LangSmith.

Fluxo:
  1. Sobe o dataset ``datasets/bug_to_user_story.jsonl`` para o LangSmith.
  2. Carrega o prompt a avaliar (pull do Hub por padrão; ``--source local``
     usa o YAML local).
  3. Executa o prompt sobre os 15 exemplos com o LLM responder (Gemini/OpenAI).
  4. Roda ``langsmith.evaluate()`` com os evaluators LLM-as-judge (Helpfulness,
     Correctness, Clarity) e os summary evaluators set-based (Precision, F1).
  5. Agrega as 5 métricas e imprime o relatório APROVADO/REPROVADO
     (critério: TODAS as métricas >= 0.8).

Uso:
    python src/evaluate.py                      # avalia v2 (pull do Hub)
    python src/evaluate.py --version v1         # avalia v1 (leonanluppi/...)
    python src/evaluate.py --source local       # usa o YAML local
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
import utils  # noqa: E402

# Rótulos amigáveis para o relatório (ordem = enunciado).
LABELS = {
    "helpfulness": "Helpfulness",
    "correctness": "Correctness",
    "f1_score": "F1-Score",
    "clarity": "Clarity",
    "precision": "Precision",
}


# ------------------------------------------------------------------------- #
# Resolução do prompt (Hub x local)
# ------------------------------------------------------------------------- #
def resolve_prompt_identifier(version: str, client=None) -> str:
    """Identificador do prompt no Hub para a versão pedida."""
    if version == "v1":
        return "leonanluppi/bug_to_user_story_v1"
    # v2 é seu: usa LANGSMITH_USERNAME ou descobre o handle automaticamente.
    username = utils.resolve_username(client)
    name = f"bug_to_user_story_{version}"
    return f"{username}/{name}" if username else name


def load_target_prompt(version: str, source: str):
    """
    Carrega o ChatPromptTemplate a avaliar.

    source="hub"   -> client.pull_prompt(identifier), com fallback para local.
    source="local" -> lê prompts/bug_to_user_story_{version}.yml.
    Retorna (prompt_obj, origem_descritiva).
    """
    local_path = utils.PROMPTS_DIR / f"bug_to_user_story_{version}.yml"

    if source == "local":
        return utils.load_chat_prompt(local_path), f"local:{local_path.name}"

    try:
        client = utils.get_langsmith_client()
        identifier = resolve_prompt_identifier(version, client)
        try:
            prompt_obj = client.pull_prompt(identifier, include_model=False)
        except TypeError:
            prompt_obj = client.pull_prompt(identifier)
        return prompt_obj, f"hub:{identifier}"
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Não foi possível fazer pull do prompt {version} do Hub ({exc}).")
        print(f"  Usando o YAML local: {local_path.name}")
        return utils.load_chat_prompt(local_path), f"local:{local_path.name}"


# ------------------------------------------------------------------------- #
# Agregação dos resultados do evaluate()
# ------------------------------------------------------------------------- #
def _row_attr(row, name):
    """Acessa campo de uma linha de resultado (dict ou objeto)."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def aggregate_report(results) -> dict:
    """
    Coleta os scores per-exemplo (helpfulness/correctness/clarity) e calcula
    localmente precision/f1 (set-based). Retorna dict {metric_key: score}.
    """
    per_metric = defaultdict(list)
    outputs, examples = [], []

    for row in results:
        run = _row_attr(row, "run")
        example = _row_attr(row, "example")
        outputs.append((getattr(run, "outputs", None) or {}))
        examples.append(example)

        eval_results = _row_attr(row, "evaluation_results") or {}
        results_list = (eval_results.get("results", [])
                        if isinstance(eval_results, dict)
                        else getattr(eval_results, "results", []))
        for er in results_list:
            key = getattr(er, "key", None)
            score = getattr(er, "score", None)
            if key is not None and score is not None:
                per_metric[key].append(score)

    report = {}
    for key in ("helpfulness", "correctness", "clarity"):
        vals = per_metric.get(key, [])
        report[key] = sum(vals) / len(vals) if vals else 0.0

    # Precision / F1 calculados localmente (garante presença no relatório).
    for m in metrics.calculate_precision_recall_f1(outputs, examples):
        if m["key"] in ("precision", "f1_score"):
            report[m["key"]] = m["score"]

    return report


# ------------------------------------------------------------------------- #
# Relatório CLI (formato do enunciado)
# ------------------------------------------------------------------------- #
def print_report(report: dict, prompt_label: str) -> bool:
    """Imprime o relatório e retorna True se APROVADO (todas as métricas >= 0.8)."""
    thr = metrics.THRESHOLD

    def line(key):
        score = report.get(key, 0.0)
        mark = "✓" if score >= thr else "✗"
        return f"  - {LABELS[key]}: {score:.2f} {mark}"

    print("=" * 50)
    print(f"Prompt: {prompt_label}")
    print("=" * 50)
    print("Métricas Derivadas:")
    for k in metrics.DERIVED_METRICS:
        print(line(k))
    print("Métricas Base:")
    for k in metrics.BASE_METRICS:
        print(line(k))

    reprovadas = [LABELS[k] for k in metrics.REPORT_METRICS
                  if report.get(k, 0.0) < thr]
    aprovado = not reprovadas
    if aprovado:
        print("✅  STATUS: APROVADO - Todas as métricas >= 0.8")
    else:
        print("❌  STATUS: REPROVADO")
        print(f"⚠   Métricas abaixo de {thr}: {', '.join(reprovadas)}")
    print(f"    (MÉDIA: {sum(report.get(k,0.0) for k in metrics.REPORT_METRICS)/len(metrics.REPORT_METRICS):.2f})")
    return aprovado


# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação de prompts no LangSmith")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"],
                        help="Versão do prompt a avaliar (default: v2)")
    parser.add_argument("--source", default="hub", choices=["hub", "local"],
                        help="Origem do prompt: Hub (pull) ou YAML local")
    parser.add_argument("--no-upload", action="store_true",
                        help="Não (re)subir o dataset ao LangSmith")
    parser.add_argument("--max-concurrency", type=int, default=2)
    args = parser.parse_args()

    from langsmith import evaluate

    print("Executando avaliação dos prompts...\n")

    # 1) Upload do dataset
    if not args.no_upload:
        try:
            n = utils.upload_langsmith_dataset(
                utils.DATASET_PATH, utils.DATASET_NAME,
                "15 bug reports -> user stories (5 simples, 7 médios, 3 complexos)")
            print(f"✓ Dataset '{utils.DATASET_NAME}' enviado ({n} exemplos).")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ Falha ao subir o dataset: {exc}", file=sys.stderr)
            print("  Verifique LANGCHAIN_API_KEY no .env.", file=sys.stderr)
            sys.exit(1)

    # 2) Carrega o prompt
    prompt_obj, prompt_label = load_target_prompt(args.version, args.source)

    # 3) LLM responder + target
    responder = utils.make_llm()

    def target_fn(inputs: dict) -> dict:
        # Robusto à chave do dataset e à variável do prompt: extrai o texto do
        # bug (bug_report/bug) e o injeta em todas as variáveis do template.
        bug_value = utils.extract_bug(inputs)
        variables = list(getattr(prompt_obj, "input_variables", []) or ["bug_report"])
        payload = {v: bug_value for v in variables}
        return utils.run_prompt(prompt_obj, responder, **payload)

    # 4) evaluate()
    try:
        results = evaluate(
            target_fn,
            data=utils.DATASET_NAME,
            evaluators=metrics.get_evaluators(),
            summary_evaluators=metrics.get_summary_evaluators(),
            experiment_prefix=f"bug_to_user_story_{args.version}",
            max_concurrency=args.max_concurrency,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Falha na avaliação: {exc}", file=sys.stderr)
        print("  Confira as credenciais (LANGCHAIN_API_KEY, GOOGLE_API_KEY/OPENAI_API_KEY).",
              file=sys.stderr)
        sys.exit(1)

    # 5) Relatório
    print()
    report = aggregate_report(results)
    aprovado = print_report(report, prompt_label)

    exp_name = getattr(results, "experiment_name", None)
    if exp_name:
        print(f"\nExperimento: {exp_name}")
    print("Dashboard: https://smith.langchain.com")

    sys.exit(0 if aprovado else 2)


if __name__ == "__main__":
    main()
