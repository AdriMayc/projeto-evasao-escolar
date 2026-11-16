# Rede Neural (MLP) para detecção de evasão — passo a passo

Documento de apoio ao notebook [`projeto_evasao_escolar_mlp.ipynb`](../projeto_evasao_escolar_mlp.ipynb). Explica, em linguagem simples, **o que é** uma rede neural, **como** ela foi montada neste projeto e **como ler** os resultados.

---

## 1. Visão geral em 30 segundos

- **Problema:** dado o perfil e o desempenho de um estudante, prever se ele vai **evadir** (`1`) ou **se graduar** (`0`).
- **Modelo:** `MLPClassifier` (*Multi-Layer Perceptron*), uma rede neural simples do `scikit-learn`.
- **Três momentos de previsão (cenários):** matrícula (24 variáveis), após o 1º semestre (30) e após o 2º semestre (36).
- **Métrica principal:** *recall* da evasão — quantos estudantes que realmente evadiram a rede conseguiu encontrar.

---

## 2. O que é uma rede neural?

Pense numa rede neural como uma **linha de montagem de decisões**, em que cada estação faz uma conta simples e passa o resultado adiante.

### 2.1 O neurônio

Um neurônio faz apenas três coisas:

1. **Recebe números** (por exemplo, nota do 1º semestre, idade, se tem bolsa).
2. **Multiplica cada número por um peso** e soma tudo (mais um valor fixo chamado *bias*). O peso diz o quanto aquela informação importa.
3. **Aplica uma função de ativação**, que decide o quanto o neurônio "acende".

```
resultado = ativação( peso1·nota + peso2·idade + peso3·bolsa + ... + bias )
```

### 2.2 As camadas

Neurônios são agrupados em camadas:

```
 Entrada            Camada oculta 1     Camada oculta 2      Saída
 (24, 30 ou 36      (64 neurônios)      (32 neurônios)       (1 neurônio)
  variáveis)
   ● ───────────────► ●  ───────────────► ●  ───────────────►  ●  → probabilidade
   ● ───────────────► ●  ───────────────► ●                       de evasão
   ● ───────────────► ●  ───────────────► ●
   ...                ...                 ...
```

- **Entrada:** uma "porta" para cada variável do estudante.
- **Camadas ocultas:** combinam as variáveis em padrões cada vez mais abstratos. Exemplo: a primeira camada pode aprender "poucas aprovações", a segunda "poucas aprovações + mensalidade atrasada".
- **Saída:** um único número entre 0 e 1 (probabilidade de evasão). Acima de 0,5 o estudante é classificado como **evasão**.

### 2.3 Por que tem função de ativação?

Sem ela, empilhar camadas seria o mesmo que uma única conta linear. A ativação **ReLU** usada aqui é muito simples: se o resultado for negativo, vira `0`; se for positivo, passa igual. Isso permite que a rede aprenda relações **não lineares** (por exemplo, "o risco só sobe forte quando a nota cai *e* as aprovações também").

Na saída é usada a função **sigmoide**, que espreme qualquer número para o intervalo 0–1, ideal para representar probabilidade.

### 2.4 Como a rede aprende?

O treinamento é um ciclo repetido muitas vezes:

1. **Palpite:** a rede recebe um estudante e faz uma previsão com os pesos atuais (no começo, aleatórios).
2. **Erro:** compara o palpite com a resposta real (evadiu ou não) e calcula o tamanho do erro (*função de perda*).
3. **Ajuste:** o algoritmo de **retropropagação** (*backpropagation*) descobre quanto cada peso contribuiu para o erro e o otimizador **Adam** corrige todos os pesos um pouquinho na direção que reduz o erro.
4. **Repetir:** percorrer todos os estudantes do treino uma vez é uma **época**. A rede faz dezenas de épocas até parar de melhorar.

Neste projeto a rede parou após **20 épocas** (matrícula), **43** (1º semestre) e **63** (2º semestre).

---

## 3. Passo a passo do notebook

### Passo 1 — Carregar e padronizar a base
Lê `data/data.csv` (4.424 estudantes), limpa os nomes das colunas e cria o dicionário `names_pt` com a tradução das variáveis para os gráficos.

### Passo 2 — Preparar o alvo e os cenários
- Remove quem está `Enrolled` (ainda cursando): não há resposta definitiva para eles.
- Converte `Dropout → 1` e `Graduate → 0`. Restam **3.630 estudantes** (1.421 evasões e 2.209 não evasões).
- Cria três conjuntos de variáveis, conforme a informação disponível em cada momento:

| Cenário | O que a rede enxerga | Nº de variáveis |
|---|---|---|
| Matrícula | Perfil, socioeconômico e ingresso | 24 |
| 1º semestre | Matrícula + desempenho do 1º semestre | 30 |
| 2º semestre | Matrícula + 1º e 2º semestres | 36 |

### Passo 3 — Análise exploratória
- **Distribuição do alvo:** mostra que há mais não evasões (≈ 61%) que evasões (≈ 39%). Esse desbalanceamento exige cuidado (ver Passo 5).
- **Correlações:** mostram quais variáveis andam junto com a evasão (aprovações e notas, por exemplo). É só uma pista: correlação não prova causa.

### Passo 4 — Separar treino e teste
```python
train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
```
- **70% treino / 30% teste:** a rede aprende no treino e é avaliada em estudantes que **nunca viu**, como numa prova com questões inéditas.
- **`stratify=y`:** mantém a mesma proporção de evasão nos dois grupos.
- **`random_state=42`:** garante que o sorteio seja sempre o mesmo, deixando o resultado reproduzível e idêntico ao do notebook original.

### Passo 5 — Padronizar as escalas (`StandardScaler`)
Variáveis têm escalas muito diferentes (idade ~20, nota ~130, taxa de inflação ~1). Redes neurais são **muito sensíveis a isso**: variáveis com números grandes dominariam o aprendizado. O `StandardScaler` transforma cada variável para média 0 e desvio 1.

> **Detalhe importante:** o scaler é ajustado (`fit`) **apenas no treino** e depois só aplicado (`transform`) no teste. Ajustar no teste "vazaria" informação do futuro para o modelo (*data leakage*).

### Passo 6 — Equilibrar as classes (sobreamostragem)
Os modelos do notebook original usam `class_weight="balanced"`, que dá mais peso aos erros da classe minoritária. O `MLPClassifier` **não tem** esse parâmetro. A alternativa usada aqui:

- A função `oversample_minority` **duplica aleatoriamente** estudantes que evadiram, **somente no conjunto de treino**, até as duas classes terem o mesmo tamanho.
- O conjunto de **teste não é tocado**, para que as métricas reflitam a realidade.

Sem isso, a rede tenderia a "chutar" sempre não evasão, pois acertaria ~61% dos casos sem aprender nada útil.

### Passo 7 — Montar a rede (`build_mlp`)

| Parâmetro | Valor | Significado simples |
|---|---|---|
| `hidden_layer_sizes` | `(64, 32)` | Duas camadas ocultas: 64 e depois 32 neurônios. Tamanho moderado para uma base pequena. |
| `activation` | `"relu"` | Função de ativação das camadas ocultas (ver 2.3). |
| `alpha` | `1e-3` | Regularização L2: penaliza pesos muito grandes, o que evita decorar o treino (*overfitting*). |
| `early_stopping` | `True` | Separa uma parte do treino para validação e **para o treino quando a rede deixa de melhorar**. |
| `validation_fraction` | `0.1` | 10% do treino usado como validação interna. |
| `n_iter_no_change` | `10` | Se passar 10 épocas sem melhora, para. |
| `max_iter` | `500` | Limite máximo de épocas (segurança). |
| `random_state` | `42` | Sorteio dos pesos iniciais reproduzível. |

Ao todo a rede tem cerca de **3.700 a 4.500 pesos** para ajustar, dependendo do cenário.

### Passo 8 — Treinar e prever
```python
model.fit(X_train_bal, y_train_bal)   # aprende
preds = model.predict(X_test_scaled)  # classifica os estudantes do teste
```
Isso é repetido **uma vez por cenário**, sempre com uma rede nova.

### Passo 9 — Calcular as métricas
Para a classe **evasão**:

- **Precision:** dos alunos que a rede marcou como risco, quantos realmente evadiram. Mede **alertas corretos**.
- **Recall:** dos alunos que realmente evadiram, quantos a rede encontrou. Mede **cobertura**.
- **F1:** média que equilibra os dois.

> Neste projeto o **recall é a métrica prioritária**: um falso negativo é um estudante em risco que ninguém acompanhou.

### Passo 10 — Matriz de confusão
Tabela 2×2 que mostra os acertos e erros de cada tipo:

|  | Previu não evasão | Previu evasão |
|---|---|---|
| **Real: não evasão** | acerto | falso positivo (alerta desnecessário) |
| **Real: evasão** | **falso negativo (risco não detectado)** | acerto |

### Passo 11 — Explicar o modelo (importância por permutação)
Redes neurais são "caixas-pretas": os pesos sozinhos não dizem quais variáveis importam. A técnica usada é a **importância por permutação**:

1. Mede o recall do modelo no teste.
2. **Embaralha** os valores de **uma** variável (quebrando a relação dela com o alvo) e mede o recall de novo.
3. **Quanto mais o recall cai, mais importante era aquela variável.**
4. Repete 10 vezes para cada variável e tira a média.

---

## 4. Por que cada escolha? (decisões de projeto)

> **Aviso honesto:** os hiperparâmetros da rede (tamanho das camadas, `alpha`, etc.) são **pontos de partida razoáveis**, não valores otimizados por busca. Eles seguem práticas comuns e foram justificados pelo tamanho da base. Uma busca sistemática (`GridSearchCV`) pode melhorar o resultado.

### Escolha do modelo

| Decisão | Por que | Alternativa descartada |
|---|---|---|
| **Rede neural (MLP)** | Foi o que se pediu para testar. Captura combinações não lineares entre variáveis (ex.: nota baixa **e** mensalidade atrasada) sem precisarmos desenhá-las à mão. | — |
| **`MLPClassifier` do scikit-learn** | A base é **tabular e pequena** (3.630 linhas): não justifica Keras/PyTorch nem GPU. Usa a mesma API (`fit`/`predict`) dos modelos do notebook original, o que permite comparar de forma direta e justa. | Keras/PyTorch: mais flexíveis, porém mais código e dependências para ganho improvável aqui. |
| **Mesmo split, cenários e métricas do original** | Só muda o modelo. Assim qualquer diferença nos números vem da rede, não de um teste diferente. | — |

### Escolha dos parâmetros da rede

| Parâmetro | Valor | Por que esse valor |
|---|---|---|
| `hidden_layer_sizes` | `(64, 32)` | Formato de **funil** (afunila da entrada até a saída): a 1ª camada combina as ~30 variáveis em muitos padrões e a 2ª os resume. Duas camadas bastam para dados tabulares; mais profundidade tende a só aumentar o risco de decorar. Com ~4 mil pesos e ~2,5 mil estudantes no treino, a rede **já é grande para a base**, por isso não aumentamos mais. |
| `activation` | `relu` | Padrão consagrado: barata de calcular e evita que o aprendizado "morra" em redes com mais de uma camada (problema do gradiente que some, comum em `tanh`/`logistic`). |
| Otimizador | `adam` (padrão) | Ajusta sozinho o tamanho do passo de cada peso, funcionando bem **sem** ajustar taxa de aprendizado. `lbfgs` seria uma opção para bases muito pequenas, mas é menos estável nessa escala e não permite `early_stopping`. |
| `alpha` | `1e-3` | Como há mais pesos do que estudantes, o risco de *overfitting* é alto. O padrão do scikit-learn é `1e-4`; usamos 10× mais para **frear pesos grandes**. Valores bem maiores (≥ 1e-1) deixariam a rede simples demais e derrubariam o recall. |
| `early_stopping=True` | ligado | Principal defesa contra decorar o treino: monitora um conjunto de validação e **interrompe quando a rede para de melhorar**. É também por isso que não precisamos escolher o nº de épocas à mão (a rede parou em 20 a 63). |
| `validation_fraction` | `0.1` | 10% é o padrão: separa dados suficientes para a validação ser confiável sem tirar demais do treino. |
| `n_iter_no_change` | `10` | "Paciência": a curva de erro oscila; esperar 10 épocas evita parar por causa de um ruído passageiro. |
| `max_iter` | `500` | Apenas teto de segurança. Nunca foi atingido, pois o *early stopping* encerra bem antes. |
| `random_state` | `42` | Reprodutibilidade (pesos iniciais e sorteios iguais a cada execução) e o mesmo valor do notebook original. |

### Escolha do pré-processamento e da avaliação

| Decisão | Por que | Alternativa descartada |
|---|---|---|
| **`StandardScaler`** | Redes aprendem por gradiente, e variáveis em escalas muito diferentes (idade ~20 vs. nota ~130) fazem as de número grande dominarem. Árvores não precisam disso; a rede **precisa**. | Treinar sem escalar: convergência lenta e resultado pior. |
| **`fit` do scaler só no treino** | Evita *data leakage*: o teste deve simular dados que ainda não existem. | Ajustar na base toda: infla as métricas artificialmente. |
| **Sobreamostragem só no treino** | O `MLPClassifier` não tem `class_weight`. Duplicar a classe rara é a forma mais simples de equilibrar e **não exige biblioteca extra**. Só no treino, porque o teste precisa refletir a proporção real (≈ 39% de evasão). | *Undersampling*: descartaria ~35% dos dados numa base já pequena. *SMOTE*: cria exemplos sintéticos, mas exige `imbalanced-learn` (não instalado) e adiciona complexidade para o MVP. |
| **Split 70/30 estratificado** | O teste de 30% (1.089 alunos) é grande o bastante para métricas estáveis. A estratificação mantém a proporção de evasão nos dois lados. É o mesmo split do notebook original. | Validação cruzada: mais robusta, mas muda a comparação com o original (listada como melhoria futura). |
| **Limiar de decisão 0,5 (padrão)** | Mantém a comparação equivalente aos outros modelos. | Reduzir o limiar (ex.: 0,4) **aumentaria o recall** à custa de mais alertas falsos; é uma alavanca futura se a coordenação preferir cobrir mais casos. |
| **Recall como métrica prioritária** | Falso negativo = aluno em risco sem acompanhamento (custo alto). Falso positivo = uma conversa desnecessária (custo baixo). | Acurácia: enganosa com classes desbalanceadas. |
| **Três cenários** | Respondem "**quando** dá para prever?": na matrícula há mais tempo para agir, mas menos precisão; após o 2º semestre há mais precisão, mas menos tempo. | — |

### Escolha da explicabilidade

| Decisão | Por que | Alternativa descartada |
|---|---|---|
| **Importância por permutação** | Funciona com **qualquer** modelo (a MLP não tem `feature_importances_`) e já vem no scikit-learn. É fácil de explicar: "embaralhei a variável e o recall caiu X". | SHAP: mais detalhado (explica cada aluno), porém mais pesado e com dependência extra; fica como evolução. |
| **`scoring="recall"`** | Mede a importância **na métrica que mais interessa** ao projeto. | Usar acurácia: poderia destacar variáveis irrelevantes para encontrar evadidos. |
| **Calculada no conjunto de teste** | Mostra o que a rede usa para **generalizar**, não o que ela decorou no treino. | Calcular no treino: favorece variáveis que só ajudam a decorar. |
| **Cenário do 1º semestre** | É o melhor equilíbrio entre antecedência e desempenho, e o mesmo usado no original. | — |

---

## 5. Resultados obtidos

Classe **evasão**, conjunto de teste (1.089 estudantes):

| Cenário | Precision | Recall | F1 |
|---|---|---|---|
| Matrícula | 0,690 | 0,700 | 0,695 |
| 1º semestre | 0,857 | 0,819 | 0,838 |
| 2º semestre | 0,914 | 0,871 | 0,892 |

**Como ler:**
- Quanto mais informação do desempenho acadêmico, melhor a previsão. Só com dados de matrícula a rede acerta cerca de 70% das evasões; após o 2º semestre, cerca de 87%.
- O **1º semestre** é um bom equilíbrio: recall de 0,82 ainda com tempo para agir.

---

## 6. Limitações e cuidados

- **Base pequena:** redes neurais costumam brilhar com muitos dados. Em dados tabulares pequenos, modelos de árvore (como o Random Forest) normalmente empatam ou superam a rede.
- **Sobreamostragem duplica casos:** pode deixar a rede ligeiramente mais confiante do que deveria nos casos de evasão repetidos.
- **Um único split:** os números dependem da divisão treino/teste sorteada. Validação cruzada daria uma estimativa mais robusta.
- **Explicabilidade:** a importância por permutação indica onde investigar, não a causa da evasão.
- **O modelo apoia, não substitui,** a avaliação humana e o contato com o estudante.

---

## 7. Como executar

```bash
source venv/bin/activate
jupyter lab projeto_evasao_escolar_mlp.ipynb
```
Execute as células em ordem. O arquivo `data/data.csv` precisa existir.

## 8. Glossário rápido

| Termo | Significado |
|---|---|
| **Peso** | Número que indica a importância de uma entrada para um neurônio. |
| **Bias** | Valor somado ao final, que desloca a decisão do neurônio. |
| **Época** | Uma passada completa pelos dados de treino. |
| **Overfitting** | Quando o modelo decora o treino e vai mal em dados novos. |
| **Data leakage** | Quando informação do teste vaza para o treino, inflando os resultados. |
| **Sobreamostragem** | Duplicar exemplos da classe rara para equilibrar as classes. |
