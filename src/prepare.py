"""Preparo canonico da tabela. TODA analise deve entrar por aqui.

Nao existe "a tabela limpa" — existe a tabela certa para cada pergunta. Este modulo
expoe as quatro visoes que a analise exigiu, cada uma com a decisao documentada.
Escolher errado ja' inverteu conclusao aqui: ver docs/METODOLOGIA.md, secao 4.
"""
import unicodedata
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAW = Path(__file__).resolve().parents[1] / "data/raw/spotify_tracks.csv"

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]


# `songwriter` e `singer-songwriter` sao O MESMO rotulo com dois nomes: as mesmas
# 1.000 faixas, 1.000 track_id em comum, ZERO exclusivos de cada lado. Ou seja: o
# dataset tem 114 rotulos e 113 generos, e 1.000 faixas sao contadas duas vezes.
# NAO colapsamos na base — a estratificacao de 1.000 por rotulo e' fato do arquivo,
# e mexer nela mudaria numeros ja' publicados. Quem CONTA genero chama `genero_canonico`.
ALIAS_GENERO = {"songwriter": "singer-songwriter"}


def genero_canonico(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa rotulos que sao o mesmo genero com nome diferente.

    USE PARA: contar generos, ranquear generos, treinar classificador de genero —
    qualquer lugar onde contar a mesma faixa duas vezes distorce.
    NAO USE PARA: reproduzir numero antigo, que foi medido com 114.
    """
    return df.assign(track_genre=df.track_genre.replace(ALIAS_GENERO))


def _unitario() -> Column:
    """Audio feature normalizada pela Spotify: fechada em [0, 1]."""
    return Column(float, Check.in_range(0.0, 1.0), nullable=False)


# O dominio de cada coluna, declarado uma vez so'. Antes isto vivia em dois lugares
# que podiam divergir em silencio: em prosa, numa tabela de documentacao, e como
# string de query aqui. A versao antiga era:
#   IMPOSSIVEIS = "tempo == 0 | time_signature in [0, 1] | duration_ms == 0 | loudness > 0"
# O schema cobre as mesmas quatro condicoes e mais o resto do dominio oficial —
# verificado: nas 113.550 linhas da base, reprova exatamente as mesmas 1.221 linhas.
ESQUEMA = DataFrameSchema(
    {
        "track_id":    Column(str, Check.str_length(22, 22), nullable=False),
        # os tres textuais tem 1 linha nula, e ela e' dado faltante legitimo
        "artists":     Column(str, nullable=True),
        "album_name":  Column(str, nullable=True),
        "track_name":  Column(str, nullable=True),
        "popularity":  Column(int, Check.in_range(0, 100)),
        "duration_ms": Column(int, Check.greater_than(0)),   # zero e' extracao que falhou
        "explicit":    Column(bool),
        "danceability": _unitario(), "energy": _unitario(), "speechiness": _unitario(),
        "acousticness": _unitario(), "instrumentalness": _unitario(),
        "liveness": _unitario(), "valence": _unitario(),
        "key":         Column(int, Check.in_range(-1, 11)),  # -1 = sem tonalidade detectada
        "loudness":    Column(float, Check.less_than_or_equal_to(0.0)),  # acima de 0 e' clipping
        "mode":        Column(int, Check.isin([0, 1])),
        "tempo":       Column(float, Check.greater_than(0.0)),  # BPM zero e' impossivel
        "time_signature": Column(int, Check.in_range(3, 7)),  # o cartao oficial diz 3 a 7
        "track_genre": Column(str, nullable=False),
    },
    strict=True,
    name="spotify_tracks (dominio oficial)",
)


def fora_do_dominio(df: pd.DataFrame) -> pd.Index:
    """Indices das linhas que violam `ESQUEMA`. Vazio se a tabela estiver limpa.

    Levanta se o problema for de ESTRUTURA (coluna faltando, dtype errado) em vez
    de valor: isso nao se resolve descartando linha, e passar batido seria pior que
    falhar. Falha de valor devolve indice; falha de estrutura devolve excecao.
    """
    try:
        ESQUEMA.validate(df, lazy=True)
        return pd.Index([])
    except pa.errors.SchemaErrors as erro:
        casos = erro.failure_cases
        estruturais = casos[casos["index"].isna()]
        if not estruturais.empty:
            quais = estruturais[["column", "check"]].drop_duplicates().to_string(index=False)
            raise ValueError(f"tabela fora do esquema, nao e' problema de linha:\n{quais}") from erro
        return pd.Index(casos["index"].dropna().unique())


def defeituosas(df: pd.DataFrame) -> pd.Index:
    """Linhas com falha de extracao COMPROVADA. Nao confundir com `fora_do_dominio`.

    Sao coisas diferentes, e tratar como iguais custou caro (docs/METODOLOGIA.md, secao 4):
    das 1.221 linhas fora do dominio do cartao oficial, so' 158 sao defeito. As
    outras 1.063 sao musica real que o cartao nao previa, e descarta-las enviesa
    contra `ambient`, `comedy`, `classical` e eletronica.

    O criterio aqui e' evidencia, nao documentacao:

    - `tempo == 0` (157 linhas): nas 157, `danceability`, `speechiness` e `valence`
      estao em ZERO EXATO. Isso nao e' musica estranha, e' o extrator devolvendo
      vazio. E 157 dos 163 `time_signature == 0` sao exatamente estas linhas.
    - `duration_ms == 0` (1 linha): faixa sem duracao nao existe.

    O que NAO entra, e por que:

    - `time_signature == 1` (967): todas com tempo detectado, concentradas em
      `sleep`, `comedy`, `ambient`, `classical` e `grindcore` — repertorio sem
      metrica regular. Fenomeno musical, nao defeito.
    - `time_signature == 0` com tempo (6): features inteiramente normais. Mesmo
      caso das 967.
    - `loudness > 0` (90): maximo 4,53 dB, mediana 0,55, todas com tempo e
      compasso normais, concentradas em `drum-and-bass`, `funk` e `idm`. Master
      alto e' medida real.

    O tell que separa um do outro: defeito de extracao se ESPALHA, fenomeno se
    CONCENTRA num subgrupo com sentido.
    """
    falhou = (df.tempo == 0) | (df.duration_ms == 0)
    return df.index[falhou]


def carregar(remover_impossiveis: bool = False) -> pd.DataFrame:
    """Base de tudo: CSV sem as 450 linhas 100% identicas.

    `remover_impossiveis=True` tira as 158 faixas com falha de extracao
    comprovada — ver `defeituosas`. NAO tira tudo que esta' fora do dominio do
    cartao oficial: 1.063 dessas linhas sao musica legitima.
    """
    df = pd.read_csv(RAW, index_col=0).drop_duplicates()
    if remover_impossiveis:
        df = df.drop(defeituosas(df))
    return df


def chave_normalizada(s: pd.Series) -> pd.Series:
    """Grafia canonica para casar nome de musica e de artista.

    NFKC junta caracteres que se escrevem diferente e sao o mesmo: '\u2026' vira '...'.
    O resto e' minuscula e espaco colapsado. Sem isso, '"C" IS FOR COOKIE' e
    '"C" is for Cookie' viram duas musicas distintas.
    """
    return (s.fillna("")
             .map(lambda x: unicodedata.normalize("NFKC", str(x)))
             .str.lower().str.strip().str.replace(r"\s+", " ", regex=True))


def popularidade_corrigida(df: pd.DataFrame, normalizar: bool = False) -> pd.DataFrame:
    """Anexa `popc`: a maior popularidade entre todas as versoes da mesma musica.

    Necessario porque a mesma musica aparece com varios `track_id`, e as copias
    delistadas vem com popularity 0. Sem isso, hits latinos parecem impopulares.

    `normalizar=True` casa por grafia canonica em vez de string exata. MEDIDO no
    arquivo: funde 137 grupos a mais e muda `popc` em 296 faixas (0,261%) — dessas,
    84 zeradas so' tem o popc corrigido para cima, e 33 sao RESGATADAS de fato
    (popc 0 -> >0), isto e', deixavam de ser lidas como impopulares.

    PADRAO E' False, e a razao e' de processo, nao tecnica: ligar muda dezenas de
    saidas e todo numero ja' publicado. Ganho de 0,26% das linhas nao paga
    re-auditar tudo. Se virar padrao, roda `--conferir` e reescreve a prosa.
    """
    chave = ["track_name", "artists"]
    if normalizar:
        d = df.assign(_n=chave_normalizada(df.track_name),
                      _a=chave_normalizada(df.artists))
        melhor = d.groupby(["_n", "_a"]).popularity.max().rename("popc")
        return d.join(melhor, on=["_n", "_a"]).drop(columns=["_n", "_a"])
    melhor = df.groupby(chave).popularity.max().rename("popc")
    return df.join(melhor, on=chave)


def por_faixa_genero(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por (faixa, genero). USE PARA: qualquer analise por genero.

    NAO deduplicar entre generos: uma faixa pertence legitimamente a varios.
    Deduplicar antes de agrupar por genero enviesa a mediana em ate +-7 pontos.
    """
    return popularidade_corrigida(carregar() if df is None else df)


def por_faixa(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por `track_id`. USE PARA: propriedades acusticas da gravacao.

    Seguro para audio features, que sao estaveis por track_id. NAO usar quando
    popularidade for o alvo — prefira `por_musica`.

    A ordenacao inclui `track_genre` de PROPOSITO, e a razao derrubou a versao
    anterior deste docstring. 15.767 dos 16.299 track_id repetidos estao EMPATADOS
    no topo de popularity, e as copias diferem em `track_genre` — ou seja, quem
    sobrevive e' desempate puro. `kind="stable"` sozinho NAO resolve: ele garante
    mesma-entrada->mesma-saida, e nao independencia de ordem, porque estavel
    preserva a ordem de CHEGADA entre empatados. Medido: so' com `popularity`,
    ~8.800 dos 89.741 generos mudavam ao embaralhar as linhas do arquivo, com ou
    sem stable — o valor oscila com o sorteio, entao leia como ordem de grandeza,
    nao como constante. Com `["popularity", "track_genre"]` a diferenca vai a zero.
    Fixado pelo caso 7 do selftest.
    """
    d = popularidade_corrigida(carregar() if df is None else df)
    return (d.sort_values(["popularity", "track_genre"], ascending=[False, True], kind="stable")
             .drop_duplicates("track_id"))


def por_musica(df: pd.DataFrame | None = None, so_vivas: bool = True) -> pd.DataFrame:
    """Uma linha por (nome, artista). USE PARA: qualquer analise de popularidade.

    Colapsa as versoes duplicadas da mesma musica. `so_vivas=True` remove as que
    estao zeradas em todas as versoes (censura, nao popularidade baixa).
    """
    d = popularidade_corrigida(carregar() if df is None else df)
    d = (d.sort_values(["popularity", "track_id", "track_genre"], ascending=[False, True, True],
                       kind="stable")
          .drop_duplicates(["track_name", "artists"]))
    return d[d.popc > 0] if so_vivas else d


def com_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Anexa colunas que o dataset nao traz e o dominio pede. USE PARA: modelo.

    Nao entram em `carregar()` de proposito: derivada e' decisao de modelagem, e
    misturar com a base tornaria impossivel reproduzir os numeros ja' publicados.

    `instrumental` (bool, instrumentalness >= 0.5, corte do cartao oficial)
        Porque o continuo e' um espeto: 60,6% das faixas ficam abaixo de 0,001 e
        33,3% sao exatamente zero. Num kNN isso vira 54.417 faixas empatadas no
        mesmo ponto do eixo. E o continuo quase nao paga o custo: correlacao com
        acousticness +0,096 contra +0,090 do binario. Terceira casa decimal.

    `tempo_dj` (float, tempo dobrado ou dividido ate' cair em [70, 140), ou NaN)
        DJ mistura 70 BPM com 140 BPM — e' a mesma pulsacao, meio tempo contra
        tempo inteiro. O `tempo` cru trata as duas como distantes ao maximo.
        NaN nas 157 faixas com tempo 0, que e' sentinela e nao velocidade.

    `duracao_min` (float)
        `duration_ms` em minutos, escala que gente le'.
    """
    # tempo 0 e' SENTINELA de "nao medido", nao velocidade. Dobrar zero da' zero para
    # sempre, e o modelo receberia 157 faixas com "0 BPM" como se fosse medida. Vira NaN.
    t = df.tempo.where(df.tempo > 0)
    for _ in range(4):                      # 4 dobras cobrem 8,75 a 1120 BPM
        t = t.where(~(t >= 140), t / 2)
        t = t.where(~(t < 70), t * 2)
    return df.assign(
        instrumental=df.instrumentalness >= 0.5,
        tempo_dj=t,
        duracao_min=df.duration_ms / 60_000,
    )


def por_artista(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por (artista, faixa), com o campo `artists` desmembrado pelo ';'.

    31.437 strings cruas viram 29.858 artistas reais. Sem o split, "A;B" e'
    tratado como um artista distinto de "A".
    """
    d = popularidade_corrigida(carregar() if df is None else df)
    d = d[d.artists.notna()]
    d = d.assign(artista=d.artists.str.split(";")).explode("artista")
    d["artista"] = d.artista.str.strip()
    return d.drop_duplicates(["artista", "track_id"])


VISOES = {
    "por_faixa_genero": "analise por genero — nao deduplica entre generos",
    "por_faixa":        "propriedade acustica da gravacao — uma linha por track_id",
    "por_musica":       "analise de popularidade — colapsa versoes duplicadas",
    "por_artista":      "analise de artista — campo `artists` desmembrado",
}

if __name__ == "__main__":
    print(f"{'visao':<20} {'linhas':>9}   quando usar")
    print("-" * 78)
    for nome, quando in VISOES.items():
        n = len(globals()[nome]())
        print(f"{nome:<20} {n:>9,}   {quando}")
