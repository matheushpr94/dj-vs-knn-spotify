"""Recomendação por similaridade acústica (kNN). Avaliada por proxy, sem dado de usuário.

Não existe verdade de referência para "boa recomendação" aqui — não há ouvinte.
Mas se os vizinhos acústicos compartilham gênero e artista MUITO acima do acaso,
o espaço carrega significado. É isso que medimos.
"""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from prepare import carregar, por_faixa, por_musica
from cabecalho import cabecalho

F = ["danceability","energy","loudness","speechiness","acousticness",
     "instrumentalness","liveness","valence","tempo","duration_ms"]
K = 10


def avaliar(d, generos, rotulo):
    d = d[d.tempo > 0].reset_index(drop=True)
    print(f"\n{'='*66}\n{rotulo}  —  {len(d):,} itens\n{'='*66}")

    Z = StandardScaler().fit_transform(d[F])
    nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=-1).fit(Z)

    rng = np.random.default_rng(0)
    amostra = rng.choice(len(d), 4000, replace=False)
    _, viz = nn.kneighbors(Z[amostra])
    viz = viz[:, 1:]                        # descarta a propria faixa

    gs = d.track_id.map(generos).to_numpy()
    art = d.artists.fillna("").to_numpy()
    alb = d.album_name.fillna("").to_numpy()
    pop = d.popc.to_numpy()

    acerto_g = np.array([[bool(gs[q] & gs[v]) for v in linha]
                         for q, linha in zip(amostra, viz)])
    acerto_a = np.array([[art[q] == art[v] for v in linha]
                         for q, linha in zip(amostra, viz)])
    acerto_al = np.array([[alb[q] == alb[v] for v in linha]
                         for q, linha in zip(amostra, viz)])

    # baseline: pares aleatorios
    a, b = rng.choice(len(d), 40000), rng.choice(len(d), 40000)
    base_g = np.mean([bool(gs[i] & gs[j]) for i, j in zip(a, b)])
    base_a = np.mean(art[a] == art[b])

    print("=== 1. os vizinhos acústicos fazem sentido? ===")
    print(f"  compartilham gênero : {acerto_g.mean():.1%}   acaso {base_g:.2%}   "
          f"->  {acerto_g.mean()/base_g:.0f}x")
    print(f"  mesmo artista       : {acerto_a.mean():.2%}   acaso {base_a:.4%}  "
          f"->  {acerto_a.mean()/base_a:.0f}x")
    print(f"  mesmo álbum         : {acerto_al.mean():.2%}")
    print(f"\n  ao menos 1 dos {K} vizinhos com gênero em comum: "
          f"{acerto_g.any(axis=1).mean():.1%}")
    print(f"  {K} de {K} com gênero em comum                    : "
          f"{acerto_g.all(axis=1).mean():.1%}")

    print("\n=== 2. o modelo puxa para o popular? (viés de popularidade) ===")
    pop_q = pop[amostra]
    pop_v = np.array([pop[linha].mean() for linha in viz])
    print(f"  popularidade mediana das consultas  : {np.median(pop_q):.1f}")
    print(f"  popularidade mediana das sugestões  : {np.median(pop_v):.1f}")
    print(f"  correlação entre as duas            : "
          f"{pd.Series(pop_q).corr(pd.Series(pop_v)):+.3f}")
    print("  medianas quase iguais -> o modelo NAO empurra para o popular.")
    print("  mas a correlacao alta indica que vizinhanca acustica local carrega")
    print("  algum sinal de popularidade, coerente com o RF sobre `por_musica`")
    print("  (44,8% contra 25% de acaso — ver 16-recomputo.txt secao 3).")

    print("\n=== 3. exemplos concretos ===")
    escolhidos = [i for i in amostra[:400]
                  if pop[i] > 60 and len(str(d.track_name[i])) < 30][:4]
    for q in escolhidos:
        i = list(amostra).index(q)
        print(f"\n  ▸ {d.track_name[q]} — {str(d.artists[q])[:30]}  "
              f"[{sorted(gs[q])[0]}, pop {pop[q]:.0f}]")
        for v in viz[i][:4]:
            marca = "=" if gs[q] & gs[v] else " "
            print(f"      {marca} {str(d.track_name[v])[:34]:<36} "
                  f"{str(d.artists[v])[:22]:<24} [{sorted(gs[v])[0]}]")


def main() -> None:
    cabecalho(
        pergunta="Faixas acusticamente vizinhas fazem sentido como recomendacao?",
        como="kNN euclidiano sobre as features padronizadas, avaliado por proxy: o vizinho compartilha genero ou artista?",
        ler_assim="Acerto muito acima do acaso significa que a vizinhanca acustica captura algo real. Nao mede gosto — nao ha usuario no dataset.",
        piso="Probabilidade de dois track_id sorteados compartilharem genero ou artista.",
        metricas=['acaso'],
    )
    bruto = carregar()
    generos = bruto.groupby("track_id").track_genre.apply(set)
    avaliar(por_faixa().reset_index(drop=True), generos,
            "BASE ANTIGA — dedup por track_id (mantém cópias da mesma música)")
    avaliar(por_musica().reset_index(drop=True), generos,
            "BASE CORRIGIDA — dedup por música (nome + artista)")


if __name__ == "__main__":
    main()
