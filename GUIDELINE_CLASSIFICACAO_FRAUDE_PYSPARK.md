# Guideline do Projeto - Classificacao de Fraude com PySpark

## 1. Contexto do projeto

- Dataset aprovado: `synthetic_fraud_data.csv`
- Tamanho atual do arquivo: aproximadamente `2.93 GB`
- Tipo de problema escolhido: `classificacao`
- Coluna alvo: `is_fraud`
- Restricao principal: usar `PySpark` e `Spark SQL`, evitando fugir para `pandas`, `scikit-learn` ou funcoes fora do ecossistema Spark

## 2. Validacao rapida da sua ideia de balanceamento

Sua ideia faz sentido.

Se a base estiver em torno de:

- `80%` de `False`
- `20%` de `True`

e voce mantiver todos os registros de fraude e remover parte dos nao-fraude ate ficar `50/50`, entao voce vai preservar cerca de `40%` da base original:

- `20%` de fraude
- `20%` de nao-fraude

Como a base original tem `2.93 GB`, `40%` disso daria algo perto de `1.17 GB` em volume bruto.

Conclusao:

- Sim, voce pode balancear por undersampling dos `False`
- Em principio, voce continua acima do minimo de `1 GB`

Alerta importante:

- `Parquet` costuma comprimir melhor que `CSV`, entao o arquivo final em disco pode ficar menor que `1 GB`
- Para o relatorio, guarde a informacao de que a base aprovada original tem `2.93 GB`
- Se o professor cobrar o tamanho da base original escolhida, voce esta coberto
- Se ele cobrar que a base processada tambem permaneca acima de `1 GB`, vale confirmar isso antes

## 3. Ordem recomendada de execucao

Siga exatamente esta ordem para evitar retrabalho:

1. Ler o CSV grande com Spark
2. Conferir schema e tipos
3. Fazer limpeza basica e engenharia de atributos
4. Salvar uma versao tratada em `Parquet`
5. Fazer a EDA na base original tratada
6. Separar treino, validacao e teste
7. Balancear apenas o conjunto de treino
8. Treinar os 3 modelos
9. Avaliar os modelos no teste sem balancear
10. Comparar resultados e fechar conclusao

## 4. Ponto mais importante: nao balanceie a base inteira antes da avaliacao

O melhor caminho academico e tecnico e:

- usar a base original para EDA
- dividir em `train`, `validation` e `test`
- balancear somente o `train`
- manter `validation` e `test` com distribuicao original

Por que isso importa:

- Se voce balancear tudo antes do split, o modelo sera avaliado num mundo artificial
- Isso pode inflar `accuracy`, `precision`, `recall` e `f1`
- Para fraude, o ideal e testar na distribuicao real

Se o professor pedir explicitamente uma base balanceada como artefato:

- gere esse artefato
- mas deixe claro no relatorio que o conjunto de teste foi mantido na distribuicao original para avaliacao justa

## 5. Estrutura sugerida para os notebooks

Uma divisao simples e boa para apresentar:

1. `01_preprocessamento_balanceamento.ipynb`
2. `02_eda.ipynb`
3. `03_modelagem_classificacao.ipynb`

Se preferir, pode manter tudo em um notebook, mas separado por secoes bem claras.

## 6. Etapa 1 - Leitura da base e conversao para Parquet

### Objetivo

Carregar o CSV grande com Spark, revisar schema e salvar uma versao mais eficiente em `Parquet`.

### O que fazer

- subir o ambiente Spark
- ler o CSV com `header=True`
- evitar `inferSchema=True` logo de inicio se estiver muito lento
- se possivel, definir schema manualmente para as principais colunas
- revisar colunas numericas, booleanas e timestamp
- salvar a versao tratada em `Parquet`

### Exemplo base

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("fraud-classification")
    .getOrCreate()
)

df = (
    spark.read
    .option("header", True)
    .csv("./synthetic_fraud_data.csv")
)

df.createOrReplaceTempView("transactions_raw")
```

### Entregaveis desta etapa

- leitura funcionando
- `printSchema()`
- `count()`
- primeira visualizacao dos dados
- versao `Parquet` salva

### Dica se ficar dificil

Se a leitura estiver lenta:

- primeiro leia poucas linhas so para entender as colunas
- depois monte um `StructType` manual
- isso costuma ser melhor do que deixar o Spark adivinhar tipos em um CSV de quase `3 GB`

## 7. Etapa 2 - Pre-processamento   

### Objetivo

Deixar a base pronta para analise e modelagem.

### O que tratar

- converter `timestamp` para tipo temporal
- converter `amount`, `distance_from_home`, `transaction_hour` para numerico
- converter `card_present`, `high_risk_merchant`, `weekend_transaction`, `is_fraud` para formato consistente
- revisar nulos
- revisar duplicatas
- remover colunas com alto risco de ruido ou identificacao unica

### Colunas que provavelmente devem ser removidas ou usadas com muito cuidado

- `transaction_id`
- `customer_id`
- `card_number`
- `device_fingerprint`
- `ip_address`

Motivo:

- altissima cardinalidade
- podem gerar ruido
- podem criar pseudo-memorizacao
- nao agregam bem para um trabalho academico introdutorio

### Colunas com bom potencial

- `amount`
- `distance_from_home`
- `transaction_hour`
- `weekend_transaction`
- `high_risk_merchant`
- `card_present`
- `channel`
- `device`
- `merchant_category`
- `merchant_type`
- `country`
- `city_size`
- `card_type`

## 8. Etapa 3 - Feature engineering minima e segura

### Objetivo

Criar atributos mais explicativos sem depender de ferramentas fora do Spark.

### Sugestoes praticas

- extrair hora, dia, mes e talvez dia da semana de `timestamp`
- transformar booleanos em `0/1`
- manter poucas categoricas de baixa ou media cardinalidade
- evitar usar IDs como entrada do modelo

### Coluna dificil: `velocity_last_hour`

Essa coluna vem como texto parecido com dicionario Python, por exemplo:

```text
{'num_transactions': 1197, 'total_amount': 33498556.08, 'unique_merchants': 105, 'unique_countries': 12, 'max_single_amount': 1925480.63}
```

Ela nao esta em JSON valido porque usa aspas simples.

Voce tem duas alternativas:

1. Extrair os campos com `regexp_extract` no Spark SQL
2. Corrigir o texto para JSON valido e depois parsear

Se estiver apertado de tempo, extraia apenas:

- `num_transactions`
- `total_amount`
- `unique_merchants`
- `unique_countries`
- `max_single_amount`

Isso ja vira um bom ganho de sinal para o modelo.

## 9. Etapa 4 - EDA

### Objetivo

Entender a base e produzir insights antes do treinamento.

### Faca a EDA na base original tratada

Nao faca a EDA principal na base balanceada. O comportamento real da fraude aparece melhor na distribuicao original.

### Analises que valem a pena

- distribuicao da classe `is_fraud`
- fraude por `country`
- fraude por `merchant_category`
- fraude por `channel`
- fraude por `device`
- fraude por `transaction_hour`
- fraude em `weekend_transaction`
- estatisticas de `amount` por classe
- estatisticas de `distance_from_home` por classe

### Consultas SQL que fazem sentido

```sql
SELECT is_fraud, COUNT(*) AS total
FROM transactions
GROUP BY is_fraud;
```

```sql
SELECT merchant_category, is_fraud, COUNT(*) AS total
FROM transactions
GROUP BY merchant_category, is_fraud
ORDER BY total DESC;
```

```sql
SELECT transaction_hour, AVG(amount) AS avg_amount, SUM(CASE WHEN is_fraud = true THEN 1 ELSE 0 END) AS frauds
FROM transactions
GROUP BY transaction_hour
ORDER BY transaction_hour;
```

### O que comentar no texto do trabalho

- onde a fraude aparece mais
- quais atributos parecem separar melhor fraude de nao-fraude
- quais variaveis parecem ter pouco efeito
- quais campos foram removidos e por que

## 10. Etapa 5 - Split dos dados

### Objetivo

Separar treino, validacao e teste antes do balanceamento.

### Recomendacao

- `70%` treino
- `15%` validacao
- `15%` teste

Se quiser seguir mais estritamente SQL, faca um split com `rand()`:

```sql
SELECT *,
       rand(42) AS split_key
FROM features_table
```

Depois:

- `split_key < 0.70` -> treino
- `0.70 <= split_key < 0.85` -> validacao
- `split_key >= 0.85` -> teste

## 11. Etapa 6 - Balanceamento

### Objetivo

Corrigir o desbalanceamento sem usar bibliotecas externas.

### Melhor estrategia para o seu caso

Fazer `undersampling` da classe majoritaria (`is_fraud = false`) apenas no conjunto de treino.

### Fluxo recomendado

1. contar quantos registros de fraude existem no treino
2. separar treino fraude e treino nao-fraude
3. embaralhar os nao-fraude
4. limitar a quantidade de nao-fraude ao mesmo total de fraude
5. unir os dois conjuntos

### Exemplo de logica

```python
from pyspark.sql import functions as F

fraud_train = train_df.filter("is_fraud = true")
non_fraud_train = train_df.filter("is_fraud = false")

fraud_count = fraud_train.count()

balanced_non_fraud = non_fraud_train.orderBy(F.rand(seed=42)).limit(fraud_count)
balanced_train = fraud_train.unionByName(balanced_non_fraud)
```

Observacao:

- o ideal e embaralhar antes de `limit`
- o trecho acima usa uma ordem aleatoria com semente fixa

### Dica se ficar pesado

Ordenar a base toda por aleatoriedade pode custar caro.

Se isso travar:

1. faca um filtro aleatorio aproximado para reduzir a classe majoritaria
2. depois aplique o `limit` no subconjunto reduzido

### O que mostrar no trabalho

- contagem antes do balanceamento
- contagem depois do balanceamento
- justificativa de por que o balanceamento foi aplicado apenas ao treino

## 12. Etapa 7 - Salvar versoes importantes em Parquet

Salve pelo menos estas versoes:

- base original tratada
- treino balanceado
- validacao original
- teste original

Exemplo:

```python
treated_df.write.mode("overwrite").parquet("./data/parquet/treated")
balanced_train.write.mode("overwrite").parquet("./data/parquet/train_balanced")
valid_df.write.mode("overwrite").parquet("./data/parquet/validation")
test_df.write.mode("overwrite").parquet("./data/parquet/test")
```

## 13. Etapa 8 - Modelagem

### Modelos exigidos

Voce mencionou que a tarefa escolhida foi classificacao, entao o conjunto mais coerente e:

1. `Decision Tree Classifier`
2. `Logistic Regression`
3. `Rede Neural`

### Ponto que vale validar com o professor

Existe uma ambiguidade na frase "nao usar funcoes prontas".

Como o enunciado pede explicitamente treinar esses algoritmos, o caminho normal seria usar os estimadores do proprio `PySpark MLlib`.

Vale confirmar se o professor quis dizer:

- "nao use scikit-learn, pandas, imblearn etc."

ou se ele realmente quer:

- "implementar os algoritmos manualmente"

Se for a segunda opcao, a complexidade sobe muito, principalmente para rede neural.

Para um trabalho normal de Spark, a interpretacao mais razoavel costuma ser:

- usar `PySpark MLlib` para os modelos
- usar `Spark SQL` e `DataFrame API` no preprocessamento e analise

## 14. Etapa 9 - Avaliacao

Como o problema e fraude, nao pare em `accuracy`.

Voce deve olhar principalmente:

- `precision`
- `recall`
- `f1-score`
- matriz de confusao
- se permitido, `AUC`

### O que comparar

- qual modelo detecta mais fraudes
- qual modelo gera mais falsos positivos
- qual teve melhor equilibrio entre `precision` e `recall`
- qual teve melhor custo-beneficio para o problema

### Leitura recomendada dos resultados

Em fraude, muitas vezes um modelo com `accuracy` alta pode ser ruim se ele quase nunca identificar fraude.

Entao a discussao final deve focar em:

- capacidade de encontrar a classe positiva
- equilibrio entre pegar fraude real e evitar alarme falso demais

## 15. Estrutura sugerida para o relatorio/apresentacao

Use esta sequencia:

1. Introducao ao problema
2. Descricao do dataset
3. Justificativa da escolha
4. Descricao do pre-processamento
5. Tratamento do desbalanceamento
6. Conversao para Parquet
7. EDA com principais insights
8. Engenharia de atributos
9. Divisao treino/validacao/teste
10. Modelos treinados
11. Metricas e comparacao
12. Conclusao final

## 16. Checklist final do que voce precisa entregar

- dataset carregado com Spark
- schema revisado
- base tratada salva em Parquet
- desbalanceamento analisado
- balanceamento aplicado no treino
- EDA feita com consultas SQL e interpretacao
- tres modelos de classificacao treinados
- avaliacao comparativa com metricas adequadas
- conclusao justificando o melhor modelo

## 17. Onde voce pode travar e como destravar

### Caso 1 - CSV muito pesado para ler

- defina schema manual
- salve cedo em Parquet
- depois continue o projeto lendo do Parquet

### Caso 2 - Muitas colunas categoricas

- use poucas categoricas relevantes
- remova identificadores unicos
- comece com um conjunto enxuto de features

### Caso 3 - `velocity_last_hour` muito chata de parsear

- extraia so os campos principais com regex
- se ainda assim atrasar, documente a exclusao e siga com o resto

### Caso 4 - Balanceamento exato muito lento

- faca reducao aproximada da classe majoritaria
- depois refine

### Caso 5 - Rede neural ficar complicada

- valide primeiro se `PySpark MLlib` esta liberado
- se estiver, monte o pipeline com calma depois de fechar `Decision Tree` e `Logistic Regression`

## 18. Recomendacao pratica para a sua execucao

Se eu estivesse montando esse trabalho, eu faria nesta ordem:

1. converter o CSV para uma base tratada em Parquet
2. fazer EDA na base original tratada
3. criar tabela de features simples e limpa
4. dividir treino/validacao/teste
5. balancear apenas treino
6. treinar `Decision Tree`
7. treinar `Logistic Regression`
8. treinar `Rede Neural`
9. comparar metricas
10. fechar o texto do trabalho

## 19. Conclusao

Seu plano esta correto, com dois ajustes importantes:

- balancear preferencialmente apenas o treino
- usar a base original para EDA e para teste final

O caminho mais seguro para o trabalho e:

- `CSV grande -> tratamento -> Parquet -> EDA -> split -> balanceamento do treino -> modelos -> avaliacao`

Se quiser continuar, o proximo passo ideal e criar o notebook `01_preprocessamento_balanceamento.ipynb` ja com o codigo inicial em PySpark.
