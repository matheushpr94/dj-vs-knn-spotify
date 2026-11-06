"""Baseline explicito para as tarefas NAO supervisionadas.

A tabela de 'quanto sinal cada tarefa tem' trazia '—' na coluna de acaso para
metade das linhas. Aqui cada uma ganha o seu piso de comparacao.
"""
import numpy as np, pandas as pd, networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from prepare import carregar, por_faixa, por_artista
from cabecalho import cabecalho

F = ["danceability","energy","loudness","speechiness","acousticness",
     "instrumentalness","liveness","valence","tempo"]
rng = np.random.default_rng(0)


def embaralhar_colunas(X):
    """Quebra a correlacao entre colunas, preservando cada distribuicao marginal."""
    Y = X.copy()
    for j in range(Y.shape[1]):
        Y[:, j] = rng.permutation(Y[:, j])
    return Y


def main() -> None:
    cabecalho(
        pergunta="As tarefas nao supervisionadas tem sinal, ou o algoritmo sempre devolve alguma coisa?",
        como="Compara silhueta, variancia do PCA e metricas de rede contra o MESMO calculo em dado embaralhado.",
        ler_assim="K-Means sempre devolve grupos, inclusive em ruido. So' a comparacao com o embaralhado diz se ha estrutura.",
        piso="Dado com as colunas embaralhadas preserva as distribuicoes e destroi a correlacao — e' o piso certo.",
        metricas=['silhueta', 'variancia_explicada', 'transitividade', 'componente'],
    )
    d = por_faixa(carregar(remover_impossiveis=True))
    Z = StandardScaler().fit_transform(d[F])
    print(f"{len(d):,} faixas\n")

    print("=== AGRUPAMENTO: existe estrutura, ou K-Means so' fatia a nuvem? ===")
    idx = rng.choice(len(Z), 10_000, replace=False)
    Zs, Ss = Z[idx], embaralhar_colunas(Z)[idx]
    for nome, M in [("dado real", Zs), ("colunas embaralhadas", Ss)]:
        lab = KMeans(n_clusters=14, n_init=10, random_state=0).fit_predict(M)
        print(f"  silhueta k=14, {nome:<22}: {silhouette_score(M, lab):.3f}")
    lab_rand = rng.integers(0, 14, len(Zs))
    print(f"  silhueta com rotulo sorteado          : {silhouette_score(Zs, lab_rand):.3f}")

    print("\n=== REDUCAO DE DIMENSAO: PCA contra colunas independentes ===")
    real = PCA().fit(Z).explained_variance_ratio_
    emb = PCA().fit(embaralhar_colunas(Z)).explained_variance_ratio_
    print(f"  PC1+PC2 no dado real          : {real[:2].sum():.1%}")
    print(f"  PC1+PC2 com colunas embaralhadas: {emb[:2].sum():.1%}  <- o piso")
    print(f"  PCs para 80% real              : {int(np.searchsorted(real.cumsum(), .8))+1}")
    print(f"  PCs para 80% embaralhado       : {int(np.searchsorted(emb.cumsum(), .8))+1}")

    print("\n=== REDE: o componente gigante e' estrutura ou consequencia do grau? ===")
    pa = por_artista(carregar(remover_impossiveis=True))
    pares = (pa.groupby("track_id").artista.apply(list)
               .loc[lambda s: s.str.len().between(2, 12)])
    G = nx.Graph()
    for lst in pares:
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                G.add_edge(lst[i], lst[j])
    n, m = G.number_of_nodes(), G.number_of_edges()
    maior = len(max(nx.connected_components(G), key=len)) / n
    print(f"  grafo: {n:,} artistas · {m:,} colaboracoes")
    print(f"  maior componente, real           : {maior:.1%}")
    graus = [g for _, g in G.degree()]
    H = nx.configuration_model(graus, seed=0)
    H = nx.Graph(H); H.remove_edges_from(nx.selfloop_edges(H))
    maior_h = len(max(nx.connected_components(H), key=len)) / H.number_of_nodes()
    print(f"  maior componente, modelo nulo    : {maior_h:.1%}  <- mesma sequencia de graus")
    print(f"  transitividade real              : {nx.transitivity(G):.4f}")
    print(f"  transitividade do modelo nulo    : {nx.transitivity(H):.4f}")


if __name__ == "__main__":
    main()
