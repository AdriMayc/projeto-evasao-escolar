# Diretrizes do Projeto: Detecção de Evasão no Ensino Superior (Churn Acadêmico)

Este documento serve como um mapa para o desenvolvimento do seu MVP. Ele está dividido nas etapas do pipeline de Machine Learning estabelecidas no material da disciplina. 

---

## Passo 1: Preparação da Base e Definição do Alvo
**Objetivo:** Carregar os dados, isolar as classes definitivas (remover os alunos ainda matriculados) e categorizar as variáveis para comparar os 3 momentos de previsão (Matrícula, 1º Semestre, 2º Semestre).

**Prompt para o VSCode:**
> "Atue como um Cientista de Dados sênior. Tenho um arquivo chamado `data.csv` (separado por ponto e vírgula) que contém dados de estudantes (base UCI Dropout and Academic Success). A coluna alvo chama-se 'Target' e possui 3 classes: 'Dropout', 'Enrolled' e 'Graduate'.
> 
> Escreva um script em Python usando Pandas que faça o seguinte:
> 1. Carregue o arquivo `data.csv`.
> 2. Filtre o DataFrame para remover a classe 'Enrolled', mantendo apenas as trajetórias definitivas.
> 3. Crie uma nova coluna binária chamada 'Alvo', onde 'Dropout' = 1 (Evasão) e 'Graduate' = 0 (Não Evasão), e remova a coluna 'Target' original.
> 4. Crie três listas separadas de variáveis preditoras (features) baseadas no nome das colunas:
>    - `features_matricula`: colunas que NÃO contêm '1st sem' nem '2nd sem' (dados disponíveis antes das aulas começarem).
>    - `features_1o_semestre`: `features_matricula` + colunas que contêm '1st sem'.
>    - `features_2o_semestre`: `features_1o_semestre` + colunas que contêm '2nd sem'."

---

## Passo 2: Análise Exploratória de Dados (EDA)
**Objetivo:** Entender o desbalanceamento das classes e identificar quais variáveis têm maior correlação com a evasão antes de treinar o modelo.

**Prompt para o VSCode:**
> "Com base no DataFrame filtrado e binário que criamos, escreva o código Python usando Matplotlib e Seaborn para a Análise Exploratória de Dados (EDA). Preciso de:
> 1. Um gráfico de barras simples e elegante mostrando a contagem absoluta e o percentual da variável 'Alvo' (Evasão vs. Não Evasão) para evidenciar o desbalanceamento.
> 2. Uma matriz de correlação (heatmap) considerando APENAS as variáveis da lista `features_1o_semestre` com a coluna 'Alvo'. Destaque visualmente quais são as top 5 variáveis mais positivamente e negativamente correlacionadas com a evasão."

---

## Passo 3: Divisão dos Dados e Pré-processamento
**Objetivo:** Separar os dados em treino e teste mantendo a proporção do desbalanceamento (estratificação) e escalar os dados (como as notas e os dados macroeconômicos).

**Prompt para o VSCode:**
> "Agora, vamos preparar os dados para modelagem. Usando o scikit-learn, gere um código que:
> 1. Separe os dados em X (features) e y (Alvo).
> 2. Crie um dicionário com os três cenários de X (Matrícula, 1º Semestre, 2º Semestre).
> 3. Para cada cenário, faça o split de treino e teste (test_size=0.3) garantindo a estratificação (stratify=y) devido ao desbalanceamento das classes, utilizando um random_state fixo para reprodutibilidade.
> 4. Aplique o `StandardScaler` apenas nas variáveis preditoras (treine o scaler apenas no conjunto de treino e aplique no teste para evitar data leakage)."

---

## Passo 4: Treinamento de Modelos (Baseline e Ensemble)
**Objetivo:** Treinar algoritmos para os três cenários e comparar como o desempenho muda à medida que adicionamos dados acadêmicos.

**Prompt para o VSCode:**
> "Precisamos treinar modelos de classificação para avaliar o trade-off entre antecedência e precisão. Crie um script que:
> 1. Inicialize três modelos diferentes: Regressão Logística (Baseline), Árvore de Decisão e Random Forest.
> 2. Para garantir que a IA dê peso à minoria (Evasão=1), configure o parâmetro `class_weight='balanced'` nos modelos.
> 3. Crie um loop que treine e avalie cada um destes três modelos nos três conjuntos de dados diferentes (Matrícula, 1º Sem. e 2º Sem.)."

---

## Passo 5: Avaliação do Modelo (Foco no Risco)
**Objetivo:** Obter as métricas que realmente importam. Como queremos pegar o estudante em risco de sair, Recall da classe 1 (Evasão) é crucial.

**Prompt para o VSCode:**
> "Para avaliar os modelos treinados no passo anterior, crie uma função de avaliação que retorne:
> 1. A Matriz de Confusão para cada modelo.
> 2. O Relatório de Classificação (Classification Report) enfatizando Precision, Recall e F1-score.
> 3. Imprima de forma clara qual foi o Recall da classe 1 (Evasão) para os três cenários de tempo utilizando o Random Forest. Lembre-se que o objetivo de negócio é detectar o estudante em risco, então errar para falsos positivos (alerta falso) é menos grave do que errar falsos negativos (deixar o estudante evadir em silêncio)."

---

## Passo 6: Explicabilidade e Intervenção (SHAP / Feature Importance)
**Objetivo:** O modelo não deve ser uma "caixa preta". A coordenação do curso precisa saber *por que* o alerta soou para agir.

**Prompt para o VSCode:**
> "Escolhendo o cenário do 1º Semestre com o modelo Random Forest como o nosso 'Modelo Vencedor' para intervenção prática:
> 1. Escreva o código para extrair o Feature Importance desse modelo e plote um gráfico de barras com as 10 variáveis mais importantes para a decisão.
> 2. (Opcional se tiver a biblioteca instalada) Gere o código para criar um `shap.summary_plot` usando a biblioteca SHAP, mostrando como cada variável aumenta ou diminui o risco de evasão.
> 3. Adicione comentários no código explicando como esse output visual ajuda a secretaria da universidade a criar uma ação direcionada para o aluno."