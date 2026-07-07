# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

## Objetivo

Entregar um software capaz de:

1. Fazer **pull** de prompts do LangSmith Prompt Hub contendo prompts de baixa qualidade.
2. **Refatorar e otimizar** esses prompts usando técnicas avançadas de Prompt Engineering.
3. Fazer **push** dos prompts otimizados de volta ao LangSmith.
4. **Avaliar** a qualidade através de métricas customizadas (Helpfulness, Correctness, F1-Score, Clarity, Precision).
5. Atingir pontuação mínima de **0.8 (80%)** em **todas** as métricas de avaliação.

## Exemplo no CLI

### Prompt RUIM (v1) — ponto de partida (ilustrativo)

```
==================================================
Prompt: {seu_username}/bug_to_user_story_v1
==================================================
Métricas Derivadas:
  - Helpfulness: 0.45 ✗
  - Correctness: 0.52 ✗
Métricas Base:
  - F1-Score: 0.48 ✗
  - Clarity: 0.50 ✗
  - Precision: 0.46 ✗

❌  STATUS: REPROVADO
⚠   Métricas abaixo de 0.8: helpfulness, correctness, f1_score, clarity, precision
```

### Prompt OTIMIZADO (v2) — objetivo

```
# Após refatorar os prompts e fazer push
python src/push_prompts.py
# Executar avaliação
python src/evaluate.py

Executando avaliação dos prompts...
==================================================
Prompt: {seu_username}/bug_to_user_story_v2
==================================================
Métricas Derivadas:
  - Helpfulness: 0.94 ✓
  - Correctness: 0.96 ✓
Métricas Base:
  - F1-Score: 0.93 ✓
  - Clarity: 0.95 ✓
  - Precision: 0.92 ✓
✅  STATUS: APROVADO - Todas as métricas >= 0.8
```

> Observação de design: o CLI agrupa as métricas em **Derivadas** (Helpfulness,
> Correctness) e **Base** (F1-Score, Clarity, Precision).

## Tecnologias obrigatórias

- **Linguagem:** Python 3.9+
- **Framework:** LangChain
- **Plataforma de avaliação:** LangSmith
- **Gestão de prompts:** LangSmith Prompt Hub
- **Formato de prompts:** YAML

## Pacotes recomendados

```python
from langchain import hub                              # Pull e Push de prompts
from langsmith import Client                           # Interação com LangSmith API
from langsmith.evaluation import evaluate              # Avaliação de prompts
from langchain_openai import ChatOpenAI                # LLM OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # LLM Gemini
```

### OpenAI
- API Key: https://platform.openai.com/api-keys
- Modelo para responder: `gpt-4o-mini`
- Modelo para avaliação: `gpt-4o`
- Custo estimado: ~$1-5 para completar o desafio

### Gemini (modelo free)
- API Key: https://aistudio.google.com/app/apikey
- Modelo para responder: `gemini-2.5-flash`
- Modelo para avaliação: `gemini-2.5-flash`
- Limite: 15 req/min, 1500 req/dia

## Requisitos

### 1. Pull do Prompt inicial do LangSmith

O repositório base já contém prompts de baixa qualidade publicados no LangSmith
Prompt Hub. A primeira tarefa é criar o código capaz de fazer o pull desses
prompts para o ambiente local.

**Tarefas:**
1. Configurar as credenciais do LangSmith no arquivo `.env` (conforme `.env.example`).
2. Implementar o script `src/pull_prompts.py` (esqueleto já existe) que:
   - Conecta ao LangSmith usando as credenciais.
   - Faz pull do prompt: `leonanluppi/bug_to_user_story_v1`.
   - Salva o prompt localmente em `prompts/bug_to_user_story_v1.yml`.

### 2. Otimização do Prompt

**Tarefas:**
1. Analisar o prompt em `prompts/bug_to_user_story_v1.yml`.
2. Criar `prompts/bug_to_user_story_v2.yml` com a versão otimizada.
3. Aplicar **obrigatoriamente Few-shot Learning** (exemplos claros de entrada/saída) e **pelo menos uma** das técnicas adicionais:
   - **Chain of Thought (CoT):** instruir o modelo a "pensar passo a passo".
   - **Tree of Thought:** explorar múltiplos caminhos de raciocínio.
   - **Skeleton of Thought:** estruturar a resposta em etapas claras.
   - **ReAct:** raciocínio + ação para tarefas complexas.
   - **Role Prompting:** definir persona e contexto detalhado.
4. Documentar no `README.md` quais técnicas foram escolhidas e por quê.

**Requisitos do prompt otimizado:**
- Instruções claras e específicas.
- Regras explícitas de comportamento.
- Exemplos de entrada/saída (Few-shot) — **obrigatório**.
- Tratamento de edge cases.
- Uso adequado de System vs User Prompt.

### 3. Push e Avaliação

**Tarefas:**
1. Implementar `src/push_prompts.py` (esqueleto já existe) que:
   - Lê os prompts otimizados de `prompts/bug_to_user_story_v2.yml`.
   - Faz push para o LangSmith com nomes versionados: `{seu_username}/bug_to_user_story_v2`.
   - Adiciona metadados (tags, descrição, técnicas utilizadas).
2. Executar o script e verificar no dashboard do LangSmith se os prompts foram publicados.
3. Deixá-lo **público**.

### 4. Iteração

Espera-se **3-5 iterações**:
- Analisar métricas baixas e identificar problemas.
- Editar prompt, fazer push e avaliar novamente.
- Repetir até TODAS as métricas >= 0.8.

**Critério de Aprovação:**
- Helpfulness >= 0.8
- Correctness >= 0.8
- F1-Score >= 0.8
- Clarity >= 0.8
- Precision >= 0.8
- MÉDIA das 5 métricas >= 0.8

> **IMPORTANTE: TODAS as 5 métricas devem estar >= 0.8, não apenas a média!**

### 5. Testes de Validação

Editar `tests/test_prompts.py` e implementar, no mínimo, os **6 testes** abaixo usando pytest:

| Teste | O que verifica |
|---|---|
| `test_prompt_has_system_prompt` | Se o campo existe e não está vazio. |
| `test_prompt_has_role_definition` | Se o prompt define uma persona (ex.: "Você é um Product Manager"). |
| `test_prompt_mentions_format` | Se o prompt exige formato Markdown ou User Story padrão. |
| `test_prompt_has_few_shot_examples` | Se o prompt contém exemplos de entrada/saída (Few-shot). |
| `test_prompt_no_todos` | Garante que não sobrou nenhum `[TODO]` no texto. |
| `test_minimum_techniques` | Verifica (via metadados do YAML) se pelo menos 2 técnicas foram listadas. |

**Como validar:** `pytest tests/test_prompts.py`

## Estrutura obrigatória do projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example              # Template das variáveis de ambiente
├── requirements.txt          # Dependências Python
├── README.md                 # Documentação do processo
│
├── prompts/
│   ├── bug_to_user_story_v1.yml  # Prompt inicial (já incluso via pull)
│   └── bug_to_user_story_v2.yml  # Prompt otimizado (criar)
│
├── datasets/
│   └── bug_to_user_story.jsonl   # 15 exemplos de bugs (já incluso)
│
├── src/
│   ├── pull_prompts.py       # Pull do LangSmith (implementar)
│   ├── push_prompts.py       # Push ao LangSmith (implementar)
│   ├── evaluate.py           # Avaliação automática (pronto)
│   ├── metrics.py            # 5 métricas implementadas (pronto)
│   └── utils.py              # Funções auxiliares (pronto)
│
└── tests/
    └── test_prompts.py       # Testes de validação (implementar)
```

### O que você deve implementar
- `prompts/bug_to_user_story_v2.yml` — criar do zero com o prompt otimizado.
- `src/pull_prompts.py` — implementar o corpo das funções.
- `src/push_prompts.py` — implementar o corpo das funções.
- `tests/test_prompts.py` — implementar os 6 testes de validação.
- `README.md` — documentar o processo de otimização.

### O que já vem pronto (no boilerplate oficial; aqui é criado do zero)
- `src/evaluate.py` — script de avaliação completo.
- `src/metrics.py` — 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision).
- `src/utils.py` — funções auxiliares.
- `datasets/bug_to_user_story.jsonl` — dataset com 15 bugs (5 simples, 7 médios, 3 complexos).
- Suporte multi-provider (OpenAI e Gemini).

## VirtualEnv para Python

```bash
python3 -m venv venv
source venv/bin/activate          # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ordem de execução

```bash
# 1. Pull dos prompts ruins
python src/pull_prompts.py

# 2. Refatorar prompts (editar prompts/bug_to_user_story_v2.yml)

# 3. Push dos prompts otimizados
python src/push_prompts.py

# 4. Executar avaliação
python src/evaluate.py
```

## Entregável

1. **Repositório público no GitHub** (fork do repositório base) contendo:
   - Todo o código-fonte implementado.
   - `prompts/bug_to_user_story_v2.yml` 100% preenchido e funcional.
   - `README.md` atualizado.
2. **README.md** deve conter:
   - **A) "Técnicas Aplicadas (Fase 2)":** quais técnicas, justificativa e exemplos práticos de aplicação.
   - **B) "Resultados Finais":** link público do dashboard LangSmith, screenshots com notas >= 0.8, tabela comparativa v1 vs v2.
   - **C) "Como Executar":** instruções detalhadas, pré-requisitos e comandos por fase.
3. **Evidências no LangSmith:**
   - Link público (ou screenshots) do dashboard.
   - Dataset de avaliação com 15 exemplos.
   - Execuções dos prompts v2 com notas >= 0.8.
   - Tracing detalhado de pelo menos 3 exemplos.

## Dicas Finais

- Lembre-se da importância da **especificidade, contexto e persona** ao refatorar.
- Use **Few-shot** com 2-3 exemplos claros para melhorar drasticamente a performance.
- **CoT** é excelente para tarefas que exigem raciocínio complexo (como análise de bugs).
- Use o **Tracing do LangSmith** como principal ferramenta de debug.
- **Não altere os datasets** de avaliação — apenas os prompts em `bug_to_user_story_v2.yml`.
- **Itere, itere, itere** — é normal precisar de 3-5 iterações.
- **Documente o processo** — a jornada de otimização é tão importante quanto o resultado.

## Repositórios úteis

- Repositório boilerplate do desafio (template a ser forkado).
- LangSmith Documentation.
- Prompt Engineering Guide.
