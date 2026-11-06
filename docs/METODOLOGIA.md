# Metodologia

Como a análise foi feita, o fundamento de cada técnica e, principalmente, **as conclusões
que mudaram no caminho e o que as derrubou**. As correções estão aqui de propósito: são a
parte reaproveitável.

Cada número cita o arquivo de `outputs/` de onde saiu. Para conferir, rode
`uv run python src/rodar_tudo.py --conferir`.

---

## 1. Princípios que valem para toda a análise

| Princípio | Por quê | Onde aparece |
|---|---|---|
| **Piso explícito em toda saída** | número sozinho não significa nada: 35% de acerto é ótimo entre 114 classes (acaso 0,88%) e péssimo entre 2 | o campo `PISO` no cabeçalho de cada `outputs/*.txt` |
| **Holdout estratificado** | modelo avaliado no dado em que treinou mede memorização | todos os classificadores (`train_test_split(..., stratify=y)`) |
| **Padronizar antes de qualquer geometria** | `tempo` vai a 243 e `danceability` vive em [0, 1]; sem padronizar, a distância vira "diferença de BPM" | kNN, PCA, K-Means, escore de pertencimento |
| **Declarar a chave de deduplicação** | é decisão de modelagem, não de limpeza. Trocar a chave inverteu uma conclusão aqui (seção 4.1) | `src/prepare.py`, quatro visões nomeadas |
| **Nulo bem medido é entrega** | "pesar as features não paga" evita que a próxima pessoa refaça a mesma busca | `15-pesar-as-features-paga.txt` |
| **Busca e reserva separadas** | procurar parâmetro e relatar o resultado no mesmo dado infla o número | `src/p15_pesar_as_features_paga.py` |

## 2. Técnicas, e a pergunta que cada uma responde

| # | Técnica | Pergunta | Fundamento, e a alternativa descartada | Script |
|---|---|---|---|---|
| 1 | Auditoria de domínio + **schema `pandera`** | o arquivo é o que promete? | o domínio das 20 colunas é declarado uma vez em `ESQUEMA`, não em prosa | `p02`, `prepare.py` |
| 2 | **Selftest com casos ruins** | a checagem pega o que deveria pegar? | teste que só passa em dado bom não prova nada. Injeta violação coluna a coluna e exige que a falha **estrutural** (coluna renomeada) levante erro em vez de devolver lista vazia | `p05` |
| 3 | Auto-junção por (nome, artista) | o zero de `popularity` é música impopular ou cópia delistada? | se a "gêmea" viva existe e é popular, o zero é indisponibilidade | `p03`, `p09` |
| 4 | Random Forest (classificação) | o som identifica o gênero? | captura interação que a correlação par a par não vê. *Descartada:* regressão logística, porque as fronteiras dependem de combinação (`acousticness` alta é clássico **ou** folk) | `p06` |
| 5 | Árvore rasa (profundidade 3) | qual é a regra, em português? | troca acurácia por legibilidade; serve para comunicar, não para prever | `p06` |
| 6 | Random Forest (regressão) + MAE contra a mediana | dá para prever popularidade? | R² sozinho engana; o MAE na escala original contra "chutar a mediana" mostra quanto o modelo compra | `p07` |
| 7 | Pisos por embaralhamento e modelo nulo | há estrutura além do acaso? | embaralhar cada coluna em separado preserva as distribuições e destrói as relações | `p08` |
| 8 | **kNN** nas 9 audio features | o que soa parecido? | distância euclidiana padronizada. *Descartado:* cosseno, porque aqui a magnitude importa | `p10`, `p11` |
| 9 | **Regra de domínio** (Camelot + BPM ±6%) | o DJ mixaria essa transição? | 30 linhas em `camelot.py`. *Descartado:* PyCamelot, porque a tabela de 24 posições cabe auditável no repo, sem dependência | `camelot.py`, `p11` |
| 10 | Grade de parâmetros (72 combinações) | o achado depende da escolha do parâmetro? | se a recusa muda pouco ao afrouxar a regra, o achado é robusto | `p12` |
| 11 | Montador guloso com 3 critérios | as features conseguem **decidir** a ordem do set? | mesmas restrições, só muda o desempate; o acaso é o piso | `p13` |
| 12 | Escore de pertencimento (distância ao centro ÷ largura) | esta faixa cabe nesta playlist? | linha contra grupo. *Descartada:* correlação, que compara colunas, não uma linha com um grupo | `p14` |
| 13 | Ablação + 200 sorteios com **busca/reserva** | pesar as colunas melhora o escore? | o ganho que só aparece onde a busca olhou é ruído | `p15` |
| 14 | K-Means + silhueta contra piso embaralhado | quantos grupos o dado sustenta? | K-Means sempre devolve grupos; o critério é a distância ao piso, não a silhueta crua | `p16` |
| 15 | Estabilidade (ARI) + HDBSCAN | o agrupamento se repete? Existem grupos? | ARI já desconta o acaso; HDBSCAN pode responder "não há grupos" | `p17` |

## 3. Métricas, e o cuidado de cada uma

| Métrica | Leitura | Cuidado |
|---|---|---|
| acurácia top-1 / top-5 | acerto exato / resposta entre os 5 palpites | cada uma tem o seu acaso: 0,88% e 4,4% com 114 classes |
| MAE | erro médio na unidade original | só significa algo ao lado do MAE de chutar a mediana |
| R² | fração da variância explicada | 0,19 parece razoável e compra 2,1 pontos de erro (seção 4.4) |
| correlação de Spearman (posição × energia) | o set sobe de energia do começo ao fim? | +1 é arco perfeito; perto de 0 é passeio aleatório |
| silhueta | os grupos estão separados? | cresce sozinha quando `k` diminui; ler sempre contra o piso embaralhado |
| ARI | dois agrupamentos concordam? | já corrigido para o acaso: o piso é zero |

## 4. O caminho, com as correções

### 4.1 Dedup por `track_id` inverteu uma conclusão

**Primeira leitura:** faixas classificadas em 4 a 7 gêneros tinham popularidade mediana **2**,
contra 44 das que aparecem em 2 ou 3 gêneros. Parecia "música de nicho".

**O que derrubou:** os exemplos eram hits ("Ferrari", "Blinding Lights"), todos com
popularidade 0. Para cada faixa zerada, procurei uma "gêmea" com o mesmo nome e artista:
**51,8%** das zeradas desse grupo têm gêmea viva, e a gêmea tem mediana **67**
(`03-popularidade-zero-e-censura.txt`).

**Teste decisivo:** deduplicar por (nome, artista) em vez de `track_id`. O grupo de 4 a 7
gêneros vai de mediana **2 para 46**. Com isso, a conclusão se inverte: hit atravessa categoria. O que existia era
um artefato de deduplicar por `track_id` num catálogo onde a mesma música tem vários ids,
alguns delistados.

### 4.2 Desempate do dedup era sorteio

**15.767 dos 16.299** `track_id` repetidos empatam no topo de `popularity`, e as cópias diferem
em `track_genre`. Ou seja: o gênero que sobrevive ao dedup era decidido pela ordem das linhas
no arquivo. `kind="stable"` **não** resolve, porque garante "mesma entrada, mesma saída" e
não independência de ordem. Medido embaralhando o arquivo: milhares de gêneros mudavam (a ordem
de grandeza é ~8.800 de 89.741, e o número exato oscila com o sorteio). Com o desempate
explícito `["popularity", "track_genre"]` a diferença vai a **zero**. Isso está fixado pelo
caso 7 de `05-o-preparo-pega-defeito.txt`.

### 4.3 87% do que eu descartava como "impossível" era música

O cartão oficial do dataset proíbe `time_signature` = 0 ou 1 e `loudness` > 0. Isso somava 1.221
linhas fora do domínio. Perguntando **quem são** essas linhas, e não quantas:

| Regra | Linhas | O que a medição mostrou | Veredito |
|---|---|---|---|
| `tempo == 0` | 157 | `danceability`, `speechiness` e `valence` em zero exato nas 157 | defeito: o extrator devolveu vazio |
| `duration_ms == 0` | 1 | — | defeito |
| `time_signature == 1` | 967 | todas com tempo detectado, concentradas em `sleep`, `comedy`, `ambient`, `classical` | **música real**, sem métrica regular |
| `loudness > 0` | 90 | todas com tempo e compasso normais, em `drum-and-bass`, `funk`, `idm` | **medida real** de master alto |

A regra que separa um caso do outro: **defeito de extração se espalha, fenômeno se concentra**
num subgrupo com sentido. Apagar as 1.063 linhas enviesava justamente contra os gêneros que a
análise mostra serem os mais identificáveis pelo som. Agora o preparo tira só as 158 linhas
com defeito comprovado (`05-o-preparo-pega-defeito.txt`, seção 1).

### 4.4 "Prever popularidade" foi testado e rejeitado

Random Forest sobre 76.444 músicas vivas (`07-da-para-prever-popularidade.txt`):

| | |
|---|---|
| R² das audio features | 0,191 |
| MAE do modelo | 12,7 pontos |
| MAE de chutar sempre a mediana | 14,8 pontos |

Todo o esforço compra **2,1 pontos** numa escala de 0 a 100. E o alvo tem três defeitos
estruturais: mede **recência**, não sucesso (pelo cartão oficial); **14,1%** dos valores são
censura (zeros de cópias delistadas, ver `01-o-arquivo-e-o-que-promete.txt`); e **720**
`track_id` têm popularidade diferente conforme o gênero, com até **44** pontos de diferença
(`02-valores-impossiveis.txt`). Por isso `popularity` entra no produto como **filtro**, nunca como alvo.

### 4.5 O primeiro kNN estava contaminado

Com a base deduplicada por `track_id`, o kNN parecia ótimo: **6,46%** dos vizinhos eram do
mesmo artista (369 vezes o acaso). Os exemplos denunciaram o problema: a vizinha mais
próxima de "Square Hammer" era "Square Hammer", quatro vezes. Eram cópias da mesma música.
Deduplicando por música, o número honesto é **2,02%** (134 vezes o acaso)
(`10-o-knn-acha-o-que-soa-parecido.txt`). É vazamento por duplicata dos dois lados: a mesma
música como consulta e como resposta (ver [METODO.md](METODO.md), seção 3).

### 4.6 A ablação de features foi medida só na metade de busca

A primeira tabela de "quais features importam" saiu da metade de **busca**, que é o mesmo erro
que a divisão busca/reserva existe para evitar. Medida nas duas metades, a ordem das features
concorda em 2 de 9 posições, e o acaso puro já daria cerca de 1. **A ordem era ruído.** O que
sobrevive às duas metades: `tempo` não contribui para o pertencimento, embora seja a restrição
mais dura da regra do DJ (`15-pesar-as-features-paga.txt`).

### 4.7 O piso de largura escondia o sinal (registro histórico)

O escore de pertencimento divide pela largura do grupo. A primeira versão trocava largura zero
por `1e-9`. Num álbum com uma coluna constante, isso dividia por um bilionésimo e produzia
distâncias de 674 milhões de larguras, o que esmagava o tamanho de efeito para 0,36. Esses
números são da versão com o bug e não são mais reproduzidos. Com um piso que vem da escala do
dado (`PISO_LARGURA = 0.10`), o efeito real aparece (`14-esta-faixa-cabe-na-playlist.txt`).

## 5. Desafios técnicos

| Desafio | Resolução |
|---|---|
| Floresta de 114 classes × 220 árvores ocupa ~6,6 GB e é morta pelo sistema numa máquina de 8 GB | `src/floresta.py` treina em lotes e soma as probabilidades. `selftest_floresta.py` prova que o resultado é igual ao da floresta inteira (diferença máxima de 3e-16) |
| `set` em Python não tem ordem estável entre processos | toda amostra é fixada com `sorted`/`min` e semente explícita |
| MLflow exige `pandas<3` | `src/registro.py` grava cada rodada (parâmetros, métricas, commit) em `outputs/execucoes.csv` |
| saída vazia passava como "igual" no portão de reprodutibilidade | `rodar_tudo.py` trata saída vazia como falha |
