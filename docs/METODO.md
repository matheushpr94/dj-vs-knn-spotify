# Método: como conduzir uma análise de dados

Recorte do guia de método que escrevi durante a residência, com o que este projeto
usou: como escolher a família de técnica, o que conferir antes de confiar num
resultado, e as sete perguntas para as duas primeiras horas com um dataset novo.
Os exemplos vêm todos deste projeto.

## 1. Que família de técnica usar

### Primeiro: o que é um "rótulo"

**Rótulo** é a resposta certa — a coluna que você quer que o modelo aprenda a preencher
sozinho. Em inglês aparece como *label* ou *target*.

- Quer adivinhar o gênero da música? O rótulo é `track_genre`.
- Quer adivinhar a popularidade? O rótulo é `popularity`.

A mesma tabela tem vários rótulos possíveis. Qual você escolhe **é** a pergunta que
você está fazendo.

### Ter rótulo ou não ter divide o mundo em dois

| Tipo | Você tem as respostas? | O que faz |
|------|------------------------|-----------|
| **Supervisionado** | sim, em todas as linhas | mostra exemplos resolvidos, o modelo aprende o padrão |
| **Não supervisionado** | não tem resposta nenhuma | pede para o modelo achar estrutura sozinho |
| **Semi-supervisionado** | tem em poucas linhas, falta em muitas | usa as poucas rotuladas para aproveitar as muitas sem rótulo |

**Por que semi-supervisionado não serve neste projeto:** todas as 114.000 linhas têm
`track_genre` preenchido. Não existe linha sem rótulo esperando para ser aproveitada.
Semi-supervisionado é remédio para **escassez de rótulo** — rotular dado costuma ser
caro e lento, e a técnica existe para esticar o pouco que você conseguiu rotular.
Aqui não há escassez, então não há o que esticar. Usar seria complicar de graça.

### As quatro famílias

| Família | Pergunta que responde | Precisa de rótulo? | Exemplo neste projeto |
|---------|----------------------|--------------------|--------------------------|
| **Regressão** | quanto? | sim, um número | prever `popularity` (0 a 100) |
| **Classificação** | qual dos tipos? | sim, uma categoria | prever `track_genre` (114 opções) |
| **Agrupamento** | quais se parecem? | não | famílias acústicas de gênero |
| **Redução de dimensionalidade** | dá para resumir? | não | PCA nas 9 audio features |

---

### Regressão — "quanto?"

Quando o que você quer prever é um **número em escala contínua**: preço, temperatura,
tempo até falhar, popularidade de 0 a 100.

**Comece por regressão linear ou Ridge.** Motivo: você consegue *ler* o modelo. Cada
coluna tem um coeficiente que diz o quanto ela empurra o resultado para cima ou para
baixo. É rápido e serve de baseline honesto. Ridge é a linear com um freio que evita
que ela se apoie demais numa coluna só — útil quando as colunas são correlacionadas.

**Suba para Random Forest ou Gradient Boosting** se desconfiar que a relação não é uma
reta, ou que duas colunas só importam **juntas**. Custo: você perde a leitura direta
do coeficiente e ganha só a importância relativa de cada coluna.

**Métricas:** R² (que fração da variação o modelo explica) e MAE (erro médio, na unidade
original — mais fácil de explicar para quem não é da área).

**Cuidado:** R² baixo não é fracasso automático. Pode ser que o dado simplesmente não
carregue esse sinal. Foi o caso de `popularity` no 01, e reportar isso é mais defensável
que forçar um modelo bonito.

---

### Classificação — "qual dos tipos?"

Quando o que você quer prever é uma **categoria**: spam ou não, qual gênero, qual
defeito, vai cancelar ou não.

**Comece por regressão logística.** Motivo: apesar do nome, ela classifica — e continua
legível. Serve de baseline.

**Suba para Random Forest** quando houver interação entre colunas. Foi o que mudou a
leitura no 01: correlação par-a-par dizia |r| ≤ 0,19 e parecia não haver sinal, mas a
floresta acertou gênero 39 vezes melhor que o acaso. **Correlação fraca e ausência de
sinal não são a mesma coisa.**

**Métricas:** acurácia **sempre ao lado do acaso**. Com 114 classes, o acaso é 1/114 =
0,88% — então 35% é excelente. Com 2 classes, o acaso é 50% e 35% seria péssimo. O
número sozinho não significa nada.

Use também a **matriz de confusão**: ela não diz só quanto você errou, diz *o que você
trocou por o que*. No 01 ela virou o produto principal — o mapa de parentesco entre
gêneros saiu de olhar quais pares o modelo confundia.

**Cuidado:** se 95% das linhas são de uma classe só, um modelo que chuta sempre essa
classe acerta 95% e não aprendeu nada. Com classes desbalanceadas, olhe precisão e
revocação por classe.

---

### Agrupamento — "quais se parecem?"

Quando **não existe rótulo** e você quer descobrir se há grupos naturais.

| Técnica | Como escolhe o número de grupos | Quando prefere |
|---------|--------------------------------|----------------|
| **K-Means** | você escolhe o k; use silhouette para decidir | rápido, grupos aproximadamente esféricos e de tamanho parecido |
| **Hierárquico** | não precisa escolher — entrega a árvore inteira e você corta onde quiser | quando quer **ver** a estrutura, não só o resultado |
| **DBSCAN** | descobre sozinho, e marca ruído à parte | grupos de formato irregular, ou quando há pontos que não pertencem a grupo nenhum |

**Cuidado obrigatório: padronizar antes.** Todas as três usam distância. No dataset do
01, `tempo` vai a 243 e `loudness` desce a −49, enquanto `danceability` vive em [0,1].
Sem padronizar, a distância vira essencialmente "diferença de BPM" e o resto some.

**O agrupamento não valida a si mesmo.** Ele sempre devolve grupos, mesmo em dado
aleatório. Antes de acreditar, cheque se os grupos batem com algo externo — no 01, as
famílias acústicas foram conferidas contra os nomes dos gêneros, e aí ficou claro que
faziam sentido.

---

### Redução de dimensionalidade — "dá para resumir sem perder muito?"

Quando você tem muitas colunas e desconfia que elas são versões da mesma coisa.

**PCA** monta eixos novos, cada um uma combinação das colunas originais, ordenados por
quanta variação capturam. A grande vantagem: você consegue **ler as cargas** e descobrir
o que cada eixo significa. No 01, o PC1 tinha energy e loudness positivos e acousticness
negativo — ou seja, era o eixo "produzido contra orgânico". Isso é interpretação, não
chute.

Serve para: enxugar colunas correlacionadas antes de modelar, e para descobrir quantas
dimensões *reais* existem. No 01 ele **refutou** uma suspeita: achávamos que as 9 audio
features fossem 2 ou 3 eixos disfarçados, e foram precisos 5 componentes para 80%.

**t-SNE e UMAP** são para **visualizar**, e só. Eles espremem tudo em 2 dimensões de um
jeito que preserva vizinhança local mas distorce distância global. Dois grupos longe um
do outro no gráfico podem não estar longe de verdade. **Nunca alimente um modelo com a
saída deles.**

**Cuidado:** padronizar antes, pelo mesmo motivo do agrupamento.

---

### As quatro medidas que a gente confunde, e o que cada uma compara

Similaridade, correlação, pertencimento e distância **não são sinônimos**. Elas comparam
coisas diferentes, e trocar uma pela outra produz frase sem sentido — *"a correlação entre
as músicas"* é o erro mais comum, e não significa nada.

**A pergunta que separa as quatro: o que está sendo comparado com o quê?**

| Medida | Compara | Responde | Sai em |
|--------|---------|----------|--------|
| **distância** | duas **linhas** | quão longe estão | unidade do dado |
| **correlação** | duas **colunas** | sobem juntas? | −1 a +1 |
| **pertencimento** | uma **linha** contra um **grupo** | quão atípica é aqui | nº de larguras |
| **similaridade** | duas **linhas**, mas invertida | quão parecidas | 0 a 1, geralmente |

#### Distância euclidiana — "quão longe estão estas duas linhas?"

É o teorema de Pitágoras com mais de duas dimensões: eleva ao quadrado a diferença de
cada coluna, soma tudo, tira a raiz.

**Quando usar:** achar o vizinho mais próximo, agrupar, recomendar "parecido com este".

**A armadilha, e ela é grave:** a coluna de escala maior domina o resultado sozinha. No
este projeto, `tempo` vai de 0 a 250 e `energy` de 0 a 1 — sem padronizar, a distância é
essencialmente a diferença de andamento e as outras oito viram enfeite. **Padronizar não
é opcional; é o que torna a conta comparável.**

**Quando NÃO usar:** com muitas colunas (dezenas), tudo fica igualmente distante de tudo
— é a maldição da dimensionalidade. E com colunas correlacionadas, o que se repete pesa
duas vezes.

#### Correlação — "quando esta coluna sobe, a outra sobe junto?"

**Quando usar:** triagem inicial, achar redundância entre colunas, decidir o que vale
investigar.

**As três armadilhas:**
1. **Só pega relação em linha reta.** Duração e popularidade têm pico no meio — a
   correlação dá quase zero e existe relação forte. Sempre olhar o gráfico.
2. **Não é causa.** Nunca foi, nunca será.
3. **Some com confundidor.** Neste projeto, o efeito de `explicit` sobre popularidade
   desaparece ao controlar por gênero.

**Quando NÃO usar:** para comparar duas linhas. Correlação é entre colunas — dizer
"correlação entre estas duas músicas" é erro de categoria.

#### Pertencimento — "esta linha é típica deste grupo?"

Distância ao centro do grupo, dividida pela largura do grupo.

**Quando usar:** dizer se algo destoa, detectar anomalia, medir coesão de um conjunto,
avaliar se um item cabe numa coleção.

**A armadilha:** grupo estreito dá escore bom fácil demais. Álbum é mais estreito que
gênero — neste projeto o acerto foi 85,4% em álbum contra 77,5% em gênero, e isso
**não** quer dizer que álbum é melhor: quer dizer que o problema é mais fácil. **Dividir
pela largura é o que torna grupos diferentes comparáveis**, e mesmo assim o número precisa
ser lido sabendo quão estreito era o grupo.

**A armadilha técnica que já mordeu aqui:** coluna constante dentro do grupo dá largura
zero. Trocar por `1e-9` para "evitar divisão por zero" significa multiplicar por um bilhão
— produziu distâncias de 674 milhões e escondeu o resultado real. O piso tem de sair da
escala do dado.

#### Similaridade — a mesma coisa que distância, virada do avesso

Cosseno mede **ângulo** entre duas linhas, ignorando magnitude — bom para texto, onde
"documento longo" não deve parecer diferente de "documento curto sobre o mesmo assunto".
Para audio features padronizadas, magnitude importa, então distância euclidiana é a
escolha natural.

### A regra que fica para toda análise

**Escolher método sem dizer por quê não conta como escolher.** Toda vez que uma técnica
entrar numa análise, o registro precisa trazer:

| O que declarar | Exemplo deste projeto |
|---|---|
| **qual pergunta** ela responde | "esta faixa cabe neste grupo?" |
| **o que ela compara** | uma linha contra um grupo |
| **qual alternativa foi descartada, e por quê** | correlação não serve: compara colunas, não linha com grupo |
| **qual é o piso** | faixa sorteada — sem modelo, 50% |
| **qual armadilha ela traz** | grupo estreito facilita; largura zero explode |

"Usei distância euclidiana" é meia resposta. **"Usei distância euclidiana em vez de
cosseno porque magnitude importa aqui, e padronizei antes porque `tempo` domina sem isso"
é a resposta.**


## 2. Antes de confiar num resultado

Do guia de análise de dados do Google, o que mais se aplica a uma análise curta:

**Separe três estágios, e não os misture.** Validação (o dado é consistente e representa o
que eu penso?), descrição (o que ele mostra, sem julgamento) e avaliação (isso é bom ou ruim
para alguém?). Misturar os três é a origem do viés de confirmação — e faz discussão travar
sem que ninguém saiba de qual dos três estão discordando.

| Prática | Aplicação aqui |
|---------|----------------|
| Olhar **distribuição**, não só resumo | média e mediana escondem bimodalidade; foi olhando distribuição que os picos de zero apareceram |
| Olhar **exemplos crus** do que o código produziu | ver as linhas, não só a contagem. Um caso de uso (escolher música para uma discoteca) só desmoronou quando lemos os nomes das faixas |
| **Fatiar** por subgrupo | a taxa de sobrevivência à limpeza varia de 21% (`latin`) a 99% (`grindcore`) — o agregado mentia |
| Medir o mesmo fenômeno de **dois jeitos** | discordância entre os dois métodos é bug de medição ou fenômeno novo; concordância é confiança barata |
| Declarar o **filtro** e contar o que ele removeu | "removi outliers" sem número é irreproduzível |

E a frase que vale emoldurar: **o dado mais surpreendente costuma ser erro, não descoberta.**
Aconteceu duas vezes neste projeto — a anomalia dos 4 a 7 gêneros era chave de dedup
errada, e o kNN espetacular estava contaminado. Resultado surpreendente é gatilho de
auditoria, não de comemoração.

## 3. Vazamento: oito tipos, e este dataset tem campo minado para vários

Kapoor e Narayanan levantaram 294 artigos em 17 áreas com resultado inflado por vazamento e
organizaram o problema em três grupos. O nosso kNN "espetacular" era o L1.4 — e não é o
único risco aqui:

| Tipo | O que é | Risco neste dataset |
|------|---------|---------------------|
| **L1.1** sem conjunto de teste | treinar e avaliar no mesmo dado | — |
| **L1.2** pré-processamento antes do corte | imputar ou padronizar no dado inteiro antes de separar treino e teste | **não hoje** — medido: os 7 usos de `StandardScaler` são todos em análise descritiva ou não supervisionada, nenhum alimenta holdout. Vira real no dia em que uma delas ganhar `train_test_split` |
| **L1.3** seleção de atributo antes do corte | escolher coluna olhando o teste | real, se escolhermos feature por desempenho |
| **L1.4** duplicatas nos dois lados | a mesma linha em treino e teste | **aconteceu**: 24.259 `track_id` repetidos e as gêmeas delistadas põem a mesma música dos dois lados |
| **L2** atributo ilegítimo | usar coluna que não estaria disponível na hora de decidir | `popularity` como feature para explicar sucesso é circular |
| **L3.1** temporal | testar em dado anterior ao de treino | não se aplica — não há coluna de tempo |
| **L3.2** amostras não independentes | dependência entre treino e teste | **real**: a mesma faixa aparece em vários gêneros |
| **L3.3** viés amostral no teste | teste não representa a população da afirmação | **real**: estratificado em 1.000 por gênero, não é o catálogo |

O antídoto que eles propõem é declarativo, não algorítmico: para cada atributo, **argumentar
por que ele é legítimo**; para o corte, argumentar por que o teste ficou isolado em todas as
etapas; e para a avaliação, argumentar que ela representa a população sobre a qual se afirma.
Três parágrafos escritos valem mais que qualquer validação cruzada rodada no piloto automático.

## 4. Multiplicidade: 30 análises não são uma análise

Gelman e Loken mostram que o problema de comparações múltiplas aparece **mesmo rodando uma
análise só**, desde que as decisões tenham sido tomadas olhando para o dado. Cada escolha de
corte, chave ou subgrupo é uma bifurcação; o conjunto de resultados possíveis é enorme sem
ninguém procurar significância de propósito. Nós rodamos 30 análises sobre a mesma tabela,
ajustando o caminho conforme o resultado aparecia.

Isso não invalida nada — define o **status**: achado exploratório gera hipótese, não
confirmação. Confirmação exige dado que não foi usado para formular a hipótese. Antes de
apresentar qualquer número:

- Quantas variantes desta análise eu tentei antes de chegar nesta?
- A hipótese foi formulada antes ou depois de eu ver este resultado?
- Ela sobrevive num recorte que eu não usei para construí-la?

## 5. Heurística antes de modelo

A regra nº 1 do *Rules of Machine Learning* é não ter medo de entregar sem ML — heurística
simples costuma capturar boa parte do ganho — e a nº 4 manda o primeiro modelo ser simples,
com a infraestrutura certa antes do algoritmo bonito. A regra de DJ deste projeto é essa
regra confirmada no nosso dado: BPM e roda de Camelot, sem treino, decidem melhor que o kNN.

Antes de treinar qualquer coisa: escrever a heurística que um humano usaria, e medi-la. Ela
é o baseline honesto, e às vezes é a entrega.


## 6. Chegou um dataset novo. E agora?

Esta seção é para quem está olhando um arquivo pela primeira vez e não sabe por onde começar.

Três dúvidas costumam aparecer logo de cara. As respostas curtas:

| A dúvida | A resposta curta |
|----------|------------------|
| **Devo comparar as colunas entre si?** | Sim, mas não para achar correlação. É para descobrir se duas colunas são **a mesma coisa com nomes diferentes** |
| **Devo verificar se as linhas são reais?** | Sim, e o que mais aparece não é linha falsa — é **a mesma coisa repetida** sem você perceber |
| **Devo procurar viés? Como?** | Sim, e o jeito é **contar por grupo**. Viés quase sempre aparece como um número que muda muito de um grupo para outro |

Abaixo, o passo a passo. São sete perguntas, na ordem em que se pergunta, e nenhuma precisa
de modelo. Cada uma tem: o que você digita, o que aparece na tela, e o que aquilo quer dizer.

---

### 1. O que é uma linha desse arquivo?

**A pergunta, em português:** se eu apontar o dedo para uma linha qualquer, ela é *o quê*?
Uma pessoa? Uma venda? Uma música? Uma música num dia específico?

**Como você faz:** abre as primeiras linhas e lê.

```python
df.head(10)
df.shape        # (linhas, colunas)
```

**O que fazer com isso:** escreva a resposta numa frase, de verdade, no README. Se você não
conseguir escrever, não dá para continuar.

**Por que isso é o primeiro de todos:** porque tudo o que você contar depois depende disso.

**Exemplo real, deste projeto:** o arquivo tem 114.000 linhas, e todo mundo dizia "são
114 mil músicas". **Não são.** Cada linha é *uma música dentro de um gênero*. A mesma música
aparece de novo em cada gênero em que ela foi classificada — algumas aparecem 9 vezes. O
número real de músicas diferentes é bem menor.

Quem não percebeu isso passou a análise inteira contando música e falando de linha.

**O sinal de alerta:** se a sua frase precisar de um "e" no meio — *"cada linha é uma música
**e** um gênero"* — então uma coisa se repete várias vezes no arquivo. Anote isso, porque
vai importar no passo 2.

---

### 2. Existe uma coluna que deveria ser o "RG" de cada linha? Ela se repete?

**A pergunta, em português:** tem alguma coluna que serve de identificador — CPF, código do
produto, ID? Se tem, ela aparece mais de uma vez?

**Como você faz:**

```python
len(df)                        # quantas linhas
df["id"].nunique()             # quantos ids DIFERENTES
```

**O que a resposta quer dizer:**

- **Os dois números iguais:** cada linha é uma coisa diferente. Simples, siga em frente.
- **Números diferentes:** tem repetição. **Pare e descubra o motivo antes de qualquer
  coisa.** Pode ser relançamento, pode ser duas planilhas juntadas, pode ser erro de coleta.
  Cada motivo pede um tratamento diferente.

**Exemplo real:** 114.000 linhas para 89.741 IDs diferentes. Ou seja, mais de 24 mil
repetições. O motivo: a mesma música classificada em vários gêneros.

**A armadilha que quase ninguém vê, e que estragou nossa análise:**

Quando você tem repetição, em algum momento vai querer ficar com **uma linha só de cada**.
O jeito normal de fazer isso é ordenar por alguma coluna e ficar com a primeira:

```python
df.sort_values("popularidade", ascending=False).drop_duplicates("id")
```

Parece inofensivo. Mas e se as cópias tiverem **o mesmo valor** de popularidade? Aí não
existe "a primeira" — o computador escolhe uma qualquer, e pode escolher outra amanhã.

Neste projeto, **quase todas as cópias estavam empatadas**: de 16.299 IDs repetidos,
15.767 tinham empate. Ou seja, em 96% dos casos era o computador sorteando, não a gente
escolhendo. E como as cópias tinham gêneros diferentes, o gênero que sobrava mudava — e
com ele todos os nossos números.

**O que fazer:** antes de ordenar e cortar, conte quantos empates existem.

```python
# quantos ids repetidos tem empate no topo?
topo = df.groupby("id")["popularidade"].transform("max")
empatados = (df["popularidade"] == topo).groupby(df["id"]).sum()
(empatados > 1).sum()
```

Se der um número alto, **o desempate é a decisão**, não um detalhe. Escolha o critério de
propósito (o mais recente? o mais completo?) e escreva por quê.

---

### 3. Alguém sorteou essas linhas, ou alguém escolheu?

**A pergunta, em português:** esse arquivo é uma foto do mundo, ou é uma seleção que alguém
montou?

**Como você faz:** conte quantas linhas tem de cada grupo.

```python
df["categoria"].value_counts()
```

**O que a resposta quer dizer:**

- **Números diferentes entre os grupos:** parece uma amostra normal. Ainda pode ter viés,
  mas pelo menos não foi montada com régua.
- **Todos os grupos com exatamente o mesmo tamanho:** alguém montou. Isso muda tudo.

**Exemplo real:** exatamente 1.000 músicas de cada um dos 114 gêneros. Ninguém sorteia
114.000 músicas e cai em 1.000 certinhas de cada gênero — isso foi construído de propósito.

**O que isso proíbe você de dizer:** qualquer frase sobre tamanho. "O gênero X é mais
popular no Spotify" é impossível de responder com esse arquivo, porque a quantidade de cada
gênero foi decidida por quem montou, não pelo mundo.

**É aqui que mora o viés.** Viés não é um defeito escondido no meio dos
números — é normalmente uma **decisão de quem montou o arquivo**, e ela aparece quando você
conta por grupo. Outros exemplos que aparecem com o mesmo comando: só clientes que ficaram
(quem cancelou sumiu), só o último ano, só uma região.

---

### 4. O nome da coluna está dizendo a verdade?

**A pergunta, em português:** essa coluna mede o que o nome dela promete?

**Como você faz — e o truque é esse:** antes de olhar qualquer número, escreva numa folha
**o que você acha que cada coluna significa**. Depois vá conferir uma por uma. As colunas em
que você errou são as que vão te derrubar lá na frente.

Para cada coluna, três perguntas:

**a) Quem mediu isso?** Tem diferença enorme entre:
- um aparelho mediu (temperatura, duração)
- uma pessoa preencheu (categoria, nome)
- **um programa de computador chutou** (é o caso das colunas de áudio do Spotify)

Se foi um programa, o número carrega os erros desse programa, e você normalmente não tem
como saber quais são.

**b) Que valores são possíveis? Existe um valor que quer dizer "não sei"?** Alguns arquivos
usam −1 ou 999 para dizer "não medido". Se não existir esse marcador, você **nunca** vai
conseguir separar "o valor é zero" de "ninguém mediu".

**c) O que pode ter sujado essa coluna?** Alguém preencheu na mão? Veio de outra planilha?
Mudou de significado no meio do caminho?

**Exemplo real:** a coluna `popularity` parece contagem de execuções. **Não é.** É uma nota
de 0 a 100 que o Spotify calcula, e ela leva em conta **quando** a música tocou, não só
quanto. Duas músicas com o mesmo público têm notas diferentes se uma é mais antiga.

E outro: `danceability` promete dizer o quanto a música é dançante. Um estudo com 244
ouvintes de verdade mostrou que **ela não bate com o que as pessoas acham dançante**. O nome
promete uma coisa, a coluna entrega outra.

**Sobre comparar colunas:** aqui é onde isso entra, e serve para uma
coisa específica — descobrir se **duas colunas são a mesma coisa disfarçada**.

```python
df[["coluna_a", "coluna_b"]].corr()
```

Se der um valor muito alto (acima de 0,8, digamos), desconfie: pode ser que uma tenha sido
**calculada a partir da outra**. Aí usar as duas juntas num modelo é contar a mesma coisa
duas vezes.

**Exemplo real:** `energy` e `loudness` andam muito juntas. Não é coincidência — o volume é
um dos ingredientes que o programa usa para calcular a energia. Correlacionar as duas mede o
programa, não a música.

---

### 5. Achou valor esquisito? Veja **quem** são essas linhas, não quantas

Esta é a mais importante.

**A situação, em português:** você olha uma coluna e vê coisa que não devia estar ali.
Zeros demais. Um número impossível, tipo velocidade negativa. Um valor que a documentação
diz que não existe.

**O reflexo errado:** contar quantas linhas são e apagar todas.

**O que fazer em vez disso:** olhar **quais** linhas são aquelas — e principalmente, ver se
elas têm alguma coisa em comum.

```python
esquisitas = df[df["coluna"] == 0]

esquisitas["categoria"].value_counts()   # elas são de algum tipo específico?
esquisitas.head(20)                      # como estão as OUTRAS colunas dessas linhas?
```

**Duas coisas podem acontecer, e elas pedem tratamentos opostos:**

**Caso A — as linhas esquisitas estão todas juntas num tipo que faz sentido.**
Então provavelmente **não é erro**. É uma característica real daquele tipo de coisa, e o
sistema que gerou o dado só não tinha uma casinha para colocar.

*Exemplo real:* o arquivo tinha 967 músicas com "compasso 1", valor que a documentação diz
não existir. Parecia erro. Fui ver de que gêneros eram: quase todas de **ambient, comédia,
clássico e ópera**. Ou seja — música de meditação, gente falando, ópera. Música que
realmente não tem batida marcada. O programa não estava errando: estava dizendo "essa música
não tem um compasso regular", e isso é **verdade**.

Se eu tivesse apagado essas 967, teria jogado fora justamente ambient e comédia — que são,
segundo nossa própria análise, os gêneros mais fáceis de reconhecer pelo som. A "limpeza"
teria estragado o achado principal.

**Caso B — as linhas esquisitas estão espalhadas por todo lado, e várias colunas quebram
juntas na mesma linha.** Aí é erro de verdade.

*Exemplo real:* 157 músicas com andamento zero (BPM zero, impossível). Fui olhar as outras
colunas dessas 157 linhas: **três colunas diferentes estavam em zero exato nas 157**.
Dançabilidade zero, fala zero, alegria zero, tudo ao mesmo tempo. Nenhuma música do mundo é
assim. Isso é o programa tendo falhado e devolvendo tudo zerado.

**A diferença entre A e B, em uma frase:** no caso A o "erro" **fazia sentido** para aquele
tipo de música e estava agrupado. No caso B **não fazia sentido nenhum** e várias colunas
quebraram juntas.

**Quanto isso custou:** de 1.221 linhas que estávamos apagando por estarem "fora do
esperado", só **158 eram erro de verdade**. As outras 1.063 eram dado bom.

---

### 6. O que esse arquivo **não** consegue responder?

**A pergunta, em português:** que perguntas eu preciso riscar da lista antes de me empolgar?

**Como você faz:** três checagens rápidas.

```python
df.columns              # tem alguma coluna de data ou hora?
df["grupo"].value_counts()   # os grupos têm o mesmo tamanho?
(df["alvo"] == 0).mean()     # o que você quer prever tem muito zero suspeito?
```

| Se acontecer isto | Você acabou de perder |
|-------------------|----------------------|
| Não tem coluna de data | qualquer pergunta com "antes", "depois", "está aumentando", "por causa de" |
| Grupos do mesmo tamanho | qualquer pergunta sobre "qual é o maior" ou "o mais comum" |
| O que você quer prever tem monte de zero que pode ser "não medido" | prever essa coluna, até resolver o que fazer com os zeros |

**Escreva a lista no README.** Parece pessimismo e não é: é a parte que impede alguém de
prometer, na apresentação, uma coisa que o arquivo não pode entregar. E é a pergunta que
todo dataset bem documentado responde: *para que este arquivo **não** deve ser usado?*

---

### 7. Se você não pudesse programar nada, que resposta daria?

**A pergunta, em português:** qual é o chute burro?

**Como você faz:**

- Se você quer **adivinhar um número** (preço, nota, idade): chute sempre a mediana para
  todo mundo. Veja quanto você erra em média.
- Se você quer **adivinhar uma categoria** (vai cancelar ou não, que gênero é): chute sempre
  a categoria mais comum. Veja quanto acerta.

**Anote esse número antes de rodar qualquer modelo.**

**Por que isso importa tanto:** porque sem ele você não tem como saber se o seu modelo é
bom. "Acertei 35%" não quer dizer nada sozinho.

- Se tem 114 categorias possíveis, chutar acerta 0,88%. Então 35% é ótimo — quarenta vezes
  melhor que chutar.
- Se tem 2 categorias possíveis, chutar acerta 50%. Aí 35% é **pior do que não fazer nada**.

**Exemplo real:** nosso modelo de popularidade erra em média 12,7 pontos numa escala de 0 a
100. Parece razoável. Só que chutar sempre a mediana erra 14,8. Todo o trabalho comprou 2
pontos. Foi por isso que abandonamos essa pergunta — e abandonar com número na mão é
resultado, não desistência.

---

### O resumo

| # | A pergunta, em português claro | O que ela evita |
|---|-------------------------------|-----------------|
| 1 | O que é uma linha desse arquivo? | contar uma coisa achando que conta outra |
| 2 | Tem identificador? Ele repete? Tem empate? | ficar com a linha errada de cada grupo |
| 3 | Alguém sorteou ou alguém montou? | falar do mundo quando o arquivo fala de uma cota |
| 4 | O nome da coluna diz a verdade? | usar uma coluna achando que ela mede outra coisa |
| 5 | Os valores esquisitos estão juntos ou espalhados? | apagar dado bom e criar viés na limpeza |
| 6 | O que esse arquivo não responde? | prometer o que o dado não tem |
| 7 | Qual é o chute burro? | achar que 35% é bom sem saber quanto é chutar |

**A ordem importa.** Não dá para saber se uma coluna é confiável antes de saber o que é uma
linha. Não dá para decidir o que apagar antes de saber se o arquivo foi montado. Cada
resposta muda o sentido da próxima pergunta — é por isso que isto é uma sequência e não uma
lista de checagens soltas.
