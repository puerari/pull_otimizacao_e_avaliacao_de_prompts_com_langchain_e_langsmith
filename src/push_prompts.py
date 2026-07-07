"""
Push do prompt otimizado (v2) para o LangSmith Prompt Hub.

Lê ``prompts/bug_to_user_story_v2.yml``, converte em ``ChatPromptTemplate`` e
faz push para ``{handle}/bug_to_user_story_v2`` com descrição, tags (as técnicas
de ``metadata.techniques``) e, por padrão, como PÚBLICO.

O handle (``LANGSMITH_USERNAME``) é opcional: se não estiver no ``.env``, é
descoberto automaticamente via API do LangSmith (``client._get_settings().tenant_handle``).

Uso:
    python src/push_prompts.py
    python src/push_prompts.py --username seu-handle    # força o handle
    python src/push_prompts.py --private                 # não publicar
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402


def push_prompt(yaml_path, name: str, username=None, extra_tags=None,
                is_public: bool = True, client=None):
    """Publica o prompt do YAML no LangSmith Hub com descrição, tags e visibilidade."""
    config = utils.load_prompt_config(yaml_path)
    prompt_obj = utils.config_to_chat_template(config)

    meta = config.get("metadata", {}) or {}
    techniques = meta.get("techniques", []) or []
    description = meta.get("description") or "Bug report -> User Story (otimizado)"
    tags = list(dict.fromkeys([*(extra_tags or []), *techniques]))  # dedup, mantém ordem

    client = client or utils.get_langsmith_client()

    # Handle: parâmetro explícito > LANGSMITH_USERNAME > descoberta automática.
    if not username:
        username = utils.resolve_username(client)
        if username:
            print(f"→ Handle do LangSmith descoberto automaticamente: '{username}'")

    if username:
        identifier = f"{username}/{name}"
    else:
        # Sem handle: não é possível publicar como público (o LangSmith exige um
        # handle para prompts públicos). Faz push privado no workspace atual.
        identifier = name
        print("⚠ Não foi possível descobrir seu handle público do LangSmith.")
        if is_public:
            print("  Publicar como PÚBLICO exige um handle. Crie o seu em:")
            print("    https://smith.langchain.com/settings")
            print("  Fazendo push como PRIVADO por ora — defina LANGSMITH_USERNAME "
                  "(ou crie o handle) e rode novamente para tornar público.")
            is_public = False

    print(f"→ Push de '{identifier}' (público={is_public})")
    print(f"  descrição: {description}")
    print(f"  tags: {tags}")

    kwargs = dict(prompt_identifier=identifier, object=prompt_obj,
                  description=description, tags=tags, is_public=is_public)
    try:
        url = client.push_prompt(**kwargs)
    except TypeError:
        # SDK antigo sem is_public/tags — tenta a forma mínima.
        url = client.push_prompt(prompt_identifier=identifier, object=prompt_obj,
                                 description=description)
    except Exception as exc:  # noqa: BLE001
        if "Nothing to commit" in str(exc):
            print("• Nada a commitar — o prompt idêntico já está publicado.")
            return None
        raise

    print(f"✓ Publicado: {url}")
    if is_public:
        print("  O prompt está PÚBLICO. Confira no dashboard do LangSmith.")
    else:
        print("  O prompt está PRIVADO. Torne-o público no dashboard se o desafio exigir.")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Push de prompt ao LangSmith Hub")
    parser.add_argument("--yaml", default=str(utils.PROMPT_V2_PATH))
    parser.add_argument("--name", default="bug_to_user_story_v2")
    parser.add_argument("--username", default=None,
                        help="Handle do LangSmith (default: env ou descoberta automática)")
    parser.add_argument("--public", dest="is_public", action="store_true", default=True,
                        help="Publicar como público (padrão)")
    parser.add_argument("--private", dest="is_public", action="store_false",
                        help="Publicar como privado")
    args = parser.parse_args()

    try:
        push_prompt(args.yaml, args.name, username=args.username,
                    extra_tags=["v2"], is_public=args.is_public)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Falha no push: {exc}", file=sys.stderr)
        print("  Verifique LANGCHAIN_API_KEY no .env.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
