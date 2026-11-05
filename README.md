# Detecção de Evasão no Ensino Superior

MVP de Machine Learning para identificar estudantes com risco de evasão (*churn acadêmico*), usando a base pública [UCI Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

## Objetivo

Apoiar a coordenação do curso na identificação **precoce** de estudantes em risco, respondendo a duas perguntas:

1. **Quando** é possível prever a evasão? O projeto compara três momentos de previsão e mostra o trade-off entre antecedência e precisão:
   - **Matrícula**: apenas dados demográficos, socioeconômicos e de ingresso;
   - **1º semestre**: matrícula + desempenho acadêmico do 1º semestre;
   - **2º semestre**: matrícula + desempenho do 1º e 2º semestres.
2. **Por que** o aluno está em risco? A importância das variáveis (*feature importance*) do modelo indica quais fatores pesaram na decisão, para que a ação seja direcionada.

Como o objetivo de negócio é não deixar um estudante evadir em silêncio, a métrica principal é o **Recall da classe Evasão**: um falso positivo (alerta falso) é menos grave que um falso negativo.

## Estrutura do repositório

```
.
├── data/data.csv                    # Base de dados (separador ";")
├── diretrizes/diretrizes.md         # Roteiro do projeto, passo a passo do pipeline
├── projeto_evasao_escolar.ipynb     # Notebook com toda a análise e os resultados
└── README.md
```

## Dados

- 4.424 estudantes, 36 colunas (35 variáveis preditoras + `Target`).
- `Target` original: `Dropout`, `Enrolled` ou `Graduate`.
- Variáveis de matrícula (idade, curso, bolsa, situação financeira etc.), desempenho em unidades curriculares do 1º e 2º semestres e indicadores macroeconômicos.

## Metodologia

1. **Preparação do alvo**: remove `Enrolled` (trajetória ainda indefinida) e cria `Alvo` binário (`Dropout` = 1, `Graduate` = 0).
2. **Cenários**: três listas de features, separadas pelo nome das colunas (`1st sem` / `2nd sem`).
3. **EDA**: distribuição das classes (desbalanceamento) e correlação das variáveis com o alvo.
4. **Pré-processamento**: split treino/teste 70/30 estratificado (`random_state=42`); `StandardScaler` ajustado só no treino, evitando *data leakage*.
5. **Modelos**: Regressão Logística (baseline), Árvore de Decisão e Random Forest, todos com `class_weight="balanced"`.
6. **Avaliação**: matriz de confusão, classification report e Precision/Recall/F1 da classe Evasão.
7. **Explicabilidade**: *feature importance* do Random Forest no cenário do 1º semestre.

## Resultados (Random Forest, classe Evasão)

| Cenário     | Precision | Recall | F1    |
|-------------|-----------|--------|-------|
| Matrícula   | 0,708     | 0,723  | 0,715 |
| 1º semestre | 0,867     | 0,840  | 0,853 |
| 2º semestre | 0,908     | 0,885  | 0,897 |

Quanto mais dados acadêmicos disponíveis, melhor o desempenho, mas menor a antecedência. O cenário do **1º semestre** é o escolhido para intervenção prática: já tem recall alto e ainda dá tempo de agir.

## Como utilizar

Requisitos: Python 3.10+.

```bash
# 1. Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar as dependências
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Abrir o notebook
jupyter notebook projeto_evasao_escolar.ipynb
```

Para apenas executar tudo sem abrir a interface (gera uma cópia com as saídas):

```bash
jupyter nbconvert --to notebook --execute projeto_evasao_escolar.ipynb --output executado.ipynb
```

Execute as células em ordem (*Run All*). O notebook lê `data/data.csv` por caminho relativo, então deve ser aberto a partir da raiz do repositório.

O notebook já contém as saídas salvas, então os gráficos e tabelas podem ser vistos sem reexecutar.

## Limitações

- Correlação e importância de variáveis não implicam causalidade.
- O modelo apoia a priorização de ações, mas **não substitui a avaliação humana** do estudante.
- Os dados vêm de uma única instituição; a generalização para outros contextos deve ser validada.
