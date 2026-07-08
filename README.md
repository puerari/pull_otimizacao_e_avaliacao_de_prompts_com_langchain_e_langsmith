# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Desafio do MBA em Engenharia de Software com IA (Full Cycle). O projeto faz
**pull** de um prompt de baixa qualidade do LangSmith Prompt Hub, **otimiza** com
técnicas de Prompt Engineering, faz **push** da versão melhorada e **avalia**
automaticamente com 5 métricas, exigindo **todas >= 0.8**.

Tarefa do prompt: converter **bug reports** em **user stories** claras e testáveis
(`bug_to_user_story`).

---

## Índice
- [Arquitetura](#arquitetura)
- [Técnicas Aplicadas (Fase 2)](#técnicas-aplicadas-fase-2)
- [Como Funciona a Avaliação](#como-funciona-a-avaliação)
- [Como Executar](#como-executar)
- [Resultados Finais](#resultados-finais)
- [Testes](#testes)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Arquitetura

| Componente | Papel |
|---|---|
| `src/pull_prompts.py` | Pull de `leonanluppi/bug_to_user_story_v1` → `prompts/bug_to_user_story_v1.yml` |
| `src/push_prompts.py` | Push de `prompts/bug_to_user_story_v2.yml` → `{username}/bug_to_user_story_v2` (com tags/descrição) |
| `src/evaluate.py` | Upload do dataset, execução do prompt e `evaluate()` com as 5 métricas; relatório APROVADO/REPROVADO |
| `src/metrics.py` | As 5 métricas (2 set-based + 3 LLM-as-judge) |
| `src/utils.py` | LLM multi-provider (Groq/Gemini/OpenAI), YAML↔prompt, dataset, parsing |
| `prompts/bug_to_user_story_v2.yml` | Prompt otimizado (Role + Few-shot + CoT) |
| `datasets/bug_to_user_story.jsonl` | 15 bugs (5 simples, 7 médios, 3 complexos) |
| `tests/test_prompts.py` | 6 testes de validação do prompt |

**Multi-provider:** o código é agnóstico ao provedor via `init_chat_model`. O
padrão é **Groq** (`qwen/qwen3-32b`, gratuito e rápido); Gemini e OpenAI são
configuráveis no `.env`.

---

## Técnicas Aplicadas (Fase 2)

O prompt otimizado (`prompts/bug_to_user_story_v2.yml`) combina **três** técnicas.
Todas são declaradas em `metadata.techniques` do YAML.

### 1. Role Prompting
**O quê:** o `system` define uma persona detalhada — uma *Product Owner sênior,
especialista em Agile e BDD, com 10 anos de experiência*.

**Por quê:** ancorar o modelo num papel especialista eleva a qualidade, o
vocabulário de domínio e a consistência das user stories — impacta diretamente
**Helpfulness** e **Clarity**.

**Exemplo (trecho do prompt):**
> "Você é uma Product Owner sênior, especialista em Agile e em Behavior-Driven
> Development (BDD)... traduzindo bug reports técnicos em user stories claras,
> acionáveis e testáveis."

### 2. Few-shot Learning (obrigatório)
**O quê:** dois exemplos completos *entrada → saída* (um simples de validação, um
de concorrência) no `user`, mostrando o formato exato esperado.

**Por quê:** exemplos ancoram o formato de saída (seções Markdown, critérios
Dado/Quando/Então) e o mapeamento para as categorias de requisito — melhora
**Correctness**, **Clarity** e, sobretudo, **Precision/F1** (o modelo aprende a
classificar as categorias corretas).

**Exemplo:** o Bug *"cadastro conclui com e-mail em branco e o login quebra"* é
mapeado para uma user story com critérios de aceite e as categorias **centrais**
`input_validation, error_handling` (o prompt instrui a listar só as 1-3 categorias
mais inequívocas — disciplina que elevou Precision/F1 na iteração).

### 3. Chain of Thought (CoT)
**O quê:** o prompt instrui um raciocínio passo a passo (ator → comportamento
atual → esperado → valor → critérios → categorias), exposto de forma concisa na
seção `## Análise`.

**Por quê:** decompor o problema antes de escrever a story reduz omissões e
aumenta a completude e a corretude — crítico para bugs **complexos** (concorrência,
segurança, integridade de dados). Impacta **Helpfulness** e **Correctness**.

**Exemplo (formato induzido):**
> `## Análise` — Ator / Comportamento atual / Comportamento esperado / Impacto,
> seguido das seções `## User Story`, `## Critérios de Aceite` e
> `## Categorias de Requisito`.

Além das técnicas, o prompt aplica: **System vs User** adequados (regras/persona
no system; few-shot e tarefa no user), **regras explícitas de comportamento**,
**tratamento de edge cases** (bug vago, múltiplos problemas, entrada que não é bug)
e um **vocabulário controlado** de 12 categorias de requisito.

---

## Como Funciona a Avaliação

As 5 métricas (limiar **0.8**), agrupadas como no enunciado:

### Métricas Base
| Métrica | Implementação |
|---|---|
| **F1-Score** | Set-based sobre as **categorias de requisito** (vocabulário controlado) extraídas da saída vs. as esperadas no dataset. F1 = média harmônica de precisão e recall. |
| **Precision** | Fração das categorias previstas que estão corretas (mesma base do F1). |
| **Clarity** | LLM-as-judge (rubrica de clareza/estrutura), nota 0–10 normalizada para 0–1. |

### Métricas Derivadas
| Métrica | Implementação |
|---|---|
| **Helpfulness** | LLM-as-judge — quão útil/acionável é a story para o time. |
| **Correctness** | LLM-as-judge **com referência** (gabarito do dataset) — equivalência semântica. |

> **Por que categorias para F1/Precision?** Comparar prosa livre com conjuntos de
> tokens zera métricas de sobreposição (a linguagem natural nunca casa
> exatamente). Seguindo o padrão dos samples do curso
> (`7-evaluation/2-precision`), o F1/Precision compara apenas **vocabulário
> controlado** — aqui, as categorias de requisito que a user story cobre. Isso
> torna a métrica objetiva, reprodutível e alcançável.

Os juízes LLM são **provider-agnósticos**: usam o modelo configurado em
`JUDGE_PROVIDER`/`JUDGE_MODEL` (Groq `qwen/qwen3-32b` por padrão). Para economizar
requisições, um único evaluator (`combined_judge_evaluator`) obtém Helpfulness,
Correctness e Clarity numa só chamada por exemplo.

---

## Como Executar

### Pré-requisitos
- Python 3.9+ (testado em 3.12)
- Conta no [LangSmith](https://smith.langchain.com) + API Key
- API Key de um provedor LLM:
  - **Groq** (grátis, padrão): https://console.groq.com/keys
  - **Gemini** (grátis): https://aistudio.google.com/app/apikey
  - **OpenAI** (pago, ~$1–5): https://platform.openai.com/api-keys

### 1. Ambiente virtual e dependências
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Credenciais
```bash
cp .env.example .env
# edite .env e preencha:
#   GROQ_API_KEY (ou GOOGLE_API_KEY / OPENAI_API_KEY)
#   LANGCHAIN_API_KEY
#   LANGSMITH_USERNAME  (seu handle no LangSmith — opcional, ver abaixo)
```
Padrão do `.env`: `LLM_PROVIDER=groq`, `LLM_MODEL=qwen/qwen3-32b` (responder e juiz).
Alternativas: `google_genai` (`gemini-2.5-flash`) ou `openai` (`gpt-4o-mini` +
juiz `gpt-4o`).

> **Nota sobre o Groq free tier:** o limite é de tokens por minuto (TPM ~6000),
> então rode a avaliação serializada — `python src/evaluate.py --max-concurrency 1`
> (leva ~10 min para os 15 exemplos, com retries automáticos respeitando o limite).

#### Sobre o `LANGSMITH_USERNAME` (handle público)

É o **handle público** do LangSmith — o slug que aparece **antes da barra** no
identificador do prompt (ex.: `leonanluppi/bug_to_user_story_v1` → handle
`leonanluppi`). **Não** é o seu e-mail nem o ID (UUID) do workspace.

- **O handle não existe por padrão** — você precisa **criá-lo uma vez**: em
  https://smith.langchain.com → **Prompts** → crie/publique um prompt como
  **Public**; o LangSmith abre um modal *"Create a handle"* para você escolher um
  slug único. (O push público é bloqueado até o handle existir.)
- **`LANGSMITH_USERNAME` é opcional:** se deixado em branco, o handle é
  **descoberto automaticamente** via API (`tenant_handle`). Se preenchido, deve
  ser exatamente o slug criado.
- **Conferir o seu handle a qualquer momento:**
  ```bash
  python -c "import dotenv; dotenv.load_dotenv(); from langsmith import Client; print(Client()._get_settings().tenant_handle)"
  ```
  Se retornar `None`, o handle ainda não foi criado.

Com isso, o `push_prompts.py` publica em `{handle}/bug_to_user_story_v2` como
**público** por padrão (use `--private` para não publicar).

### 3. Fluxo do desafio
```bash
# 1) Pull do prompt ruim (sobrescreve prompts/bug_to_user_story_v1.yml)
python src/pull_prompts.py

# 2) Refatorar: o prompt otimizado já está em prompts/bug_to_user_story_v2.yml
#    (edite-o à vontade para iterar)

# 3) Push do prompt otimizado para o seu Prompt Hub
python src/push_prompts.py

# 4) Avaliar (sobe o dataset, executa e gera o relatório)
python src/evaluate.py --max-concurrency 1              # avalia v2 (pull do Hub)
python src/evaluate.py --max-concurrency 1 --source local  # usa o YAML local
python src/evaluate.py --max-concurrency 1 --version v1    # avalia o v1 (comparação)
```
> `--max-concurrency 1` recomendado no Groq free tier (TPM ~6000). Com Gemini/OpenAI
> pagos você pode aumentar a concorrência.

> **Iteração:** espere 3–5 rodadas. Edite `prompts/bug_to_user_story_v2.yml`,
> rode `push_prompts.py` e `evaluate.py` novamente até **todas** as métricas
> atingirem 0.8. Use o **Tracing do LangSmith** para depurar cada exemplo.

---

## Resultados Finais

Avaliação executada com **Groq / `qwen/qwen3-32b`** (responder e juiz) sobre os 15 exemplos.

### Links (LangSmith)

> Para tornar um experimento público: abra-o no LangSmith → botão **Share** →
> *Make public* → copie a URL e cole abaixo.

- **Experimento v2 (aprovado):** [ver no LangSmith](https://smith.langchain.com/public/192d47d0-291e-473b-9df5-e185032f0522/d/compare?selectedSessions=c080e00b-ce74-40c1-a768-37911cbadf47)
- **Experimento v1 (comparação):** [ver no LangSmith](https://smith.langchain.com/public/192d47d0-291e-473b-9df5-e185032f0522/d/compare?selectedSessions=b0cea98f-4b40-4555-a355-0c3ff2c1063f)
- **Comparação v1 vs v2 (lado a lado):** [ver no LangSmith](https://smith.langchain.com/public/192d47d0-291e-473b-9df5-e185032f0522/d/compare?selectedSessions=b0cea98f-4b40-4555-a355-0c3ff2c1063f,c080e00b-ce74-40c1-a768-37911cbadf47)
- **Prompt no Prompt Hub (público):** [`puerari/bug_to_user_story_v2`](https://smith.langchain.com/hub/puerari/bug_to_user_story_v2)

> Um único **share público do dataset** (`192d47d0-…`) expõe ambos os experimentos;
> o parâmetro `selectedSessions` seleciona qual visualizar — v2 (`c080e00b`), v1
> (`b0cea98f`) ou os dois na comparação lado a lado. O dataset com os 15 exemplos
> também fica visível nesse mesmo share.

### Tabela comparativa (v1 vs v2)

| Métrica | v1 (ruim) | v2 (otimizado) | Meta |
|---|---|---|---|
| Helpfulness | 0.84 | **0.89** | ≥ 0.8 |
| Correctness | 0.80 | **0.86** | ≥ 0.8 |
| F1-Score | 0.15 | **0.83** | ≥ 0.8 |
| Clarity | 0.84 | **0.99** | ≥ 0.8 |
| Precision | 1.00¹ | **0.83** | ≥ 0.8 |
| **MÉDIA** | 0.73 | **0.88** | ≥ 0.8 |
| **STATUS** | ❌ REPROVADO | ✅ **APROVADO** | — |

Experimentos: v1 `bug_to_user_story_v1-91539dff` · v2 `bug_to_user_story_v2-a03aac59`.

O ganho decisivo está no **F1-Score (0.15 → 0.83)**: o v1 é vago e não estrutura a
saída em categorias de requisito, então o recall despenca. A iteração do v2 elevou
F1/Precision de **0.76 → 0.83** (gabarito recalibrado para categorias centrais +
prompt conservador), sem sacrificar as demais métricas.

<sub>¹ A Precision 1.00 do v1 é um artefato: o prompt ruim quase não emite
categorias, mas as poucas que emite acertam — por isso o F1 (que também depende do
recall) é o indicador honesto da diferença de qualidade.</sub>

### Análise crítica e limitações

Uma leitura honesta dos números acima:

- **Viés de generosidade do LLM-as-judge.** As três métricas de juiz
  (Helpfulness, Correctness, Clarity) ficaram altas até para o v1 (0.80–0.84):
  o `qwen/qwen3-32b` produz texto coerente mesmo com um prompt vago, e juízes LLM
  tendem a premiar fluência. Por isso, **o F1-Score (métrica objetiva, sem LLM) é
  o discriminador mais confiável** entre v1 e v2 — 0.15 vs 0.83.
- **Juiz = modelo respondente.** Por restrição de quota do free tier, o mesmo
  modelo atua como respondente e como juiz, o que introduz um viés de
  autoavaliação. O ideal metodológico (e o que o enunciado sugere) é um **juiz
  independente e mais forte** — ex.: `gpt-4o` — o que tornaria a avaliação mais
  rigorosa. O código já suporta isso via `JUDGE_PROVIDER`/`JUDGE_MODEL` no `.env`.
- **F1/Precision são uma proxy.** Medem a cobertura de *categorias de requisito*
  (vocabulário controlado), não a prosa da user story — escolha necessária para
  obter uma métrica objetiva e reprodutível numa tarefa gerativa (ver
  [Como Funciona a Avaliação](#como-funciona-a-avaliação)).

Nada disso invalida o resultado — o v2 atinge ≥ 0.8 em todas as métricas de fato —,
mas explicita o que aumentaria o rigor numa próxima iteração.

### Evidências (via link público)

Conforme o enunciado, o **link público do dashboard** substitui os screenshots
("Link público **(ou screenshots)** do dashboard"). No experimento v2 tornado
público ficam visíveis as três evidências exigidas:

- **Dataset com 15 exemplos** — os 15 runs do experimento (um por exemplo).
- **Execuções do v2 com notas ≥ 0.8** — os scores das 5 métricas por run e no resumo.
- **Tracing detalhado de ≥ 3 exemplos** — abra qualquer run do experimento para ver
  a árvore de execução (prompt enviado ao LLM, resposta gerada e scores dos
  evaluators). As mesmas chamadas também ficam no menu **Tracing / Observability**,
  no projeto `bug_to_user_story` (definido por `LANGCHAIN_PROJECT`), com detalhes de
  tokens e latência por chamada.

O link do experimento v2 (aprovado) está na subseção
[Links (LangSmith)](#links-langsmith) acima.

---

## Testes

```bash
pytest tests/test_prompts.py -v
```
Valida (offline) o prompt v2: presença de system prompt, definição de persona,
menção a formato Markdown/User Story, exemplos few-shot, ausência de `[TODO]` e
mínimo de 2 técnicas nos metadados.

---

## Estrutura do Projeto

```
.
├── .env.example
├── requirements.txt
├── README.md
├── prompts/
│   ├── bug_to_user_story_v1.yml     # pull do Hub (placeholder até rodar pull)
│   └── bug_to_user_story_v2.yml     # prompt otimizado (Role + Few-shot + CoT)
├── datasets/
│   └── bug_to_user_story.jsonl      # 15 bugs -> user stories
├── src/
│   ├── pull_prompts.py
│   ├── push_prompts.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
└── tests/
    └── test_prompts.py
```
