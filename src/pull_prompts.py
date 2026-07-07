"""
Pull do prompt inicial (baixa qualidade) do LangSmith Prompt Hub.

Conecta ao LangSmith com as credenciais do .env, faz pull de
``leonanluppi/bug_to_user_story_v1`` e salva localmente em
``prompts/bug_to_user_story_v1.yml`` no formato ``messages: [{role, content}]``.

Uso:
    python src/pull_prompts.py
    python src/pull_prompts.py --prompt leonanluppi/bug_to_user_story_v1 \
                               --out prompts/bug_to_user_story_v1.yml
"""
import argparse
import sys
from pathlib import Path

# Permite `import utils` ao rodar `python src/pull_prompts.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

DEFAULT_PROMPT = "leonanluppi/bug_to_user_story_v1"


def pull_prompt(prompt_identifier: str, out_path: Path) -> dict:
    """Faz pull do prompt e serializa para YAML. Retorna o dict salvo."""
    client = utils.get_langsmith_client()
    print(f"→ Conectando ao LangSmith e fazendo pull de '{prompt_identifier}'...")

    # include_model=False garante que vem apenas o template (sem o modelo anexado).
    try:
        prompt_obj = client.pull_prompt(prompt_identifier, include_model=False)
    except TypeError:
        prompt_obj = client.pull_prompt(prompt_identifier)

    config = utils.chat_template_to_config(prompt_obj)
    utils.save_prompt_config(config, out_path)

    print(f"✓ Prompt salvo em {out_path}")
    print(f"  input_variables: {config['input_variables']}")
    print(f"  mensagens: {[m['role'] for m in config['messages']]}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull de prompt do LangSmith Hub")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help=f"Identificador do prompt (default: {DEFAULT_PROMPT})")
    parser.add_argument("--out", default=str(utils.PROMPT_V1_PATH),
                        help="Caminho de saída do YAML")
    args = parser.parse_args()

    try:
        pull_prompt(args.prompt, Path(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Falha no pull: {exc}", file=sys.stderr)
        print("  Verifique LANGCHAIN_API_KEY no .env e a conexão de rede.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
