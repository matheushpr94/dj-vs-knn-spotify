"""Baixa o Spotify Tracks Dataset e confere o sha256 antes de qualquer analise.

O CSV nao e' redistribuido neste repositorio (ver "Dados e licenca" no README).
Este script busca a copia publica no Hugging Face e so' aceita o arquivo se o
hash bater com o registrado abaixo: hash diferente significa outro arquivo, e
todos os numeros do README deixariam de valer.

    uv run python baixar_dados.py              # baixa se faltar, confere sempre
    uv run python baixar_dados.py --verificar  # so' confere, nao baixa

So' usa a biblioteca padrao. Codigo de saida 0 se o arquivo confere, 1 se nao.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset"
       "/resolve/main/dataset.csv")
SHA256 = "b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd"
BYTES = 20_118_244
LINHAS = 114_000  # sem o cabecalho
DESTINO = Path(__file__).resolve().parent / "data" / "raw" / "spotify_tracks.csv"


def sha256_de(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for pedaco in iter(lambda: f.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def baixar() -> None:
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    tmp = DESTINO.with_suffix(".parcial")
    print(f"baixando {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r, tmp.open("wb") as f:
        while pedaco := r.read(1 << 20):
            f.write(pedaco)
    tmp.replace(DESTINO)


def main() -> int:
    if not DESTINO.exists():
        if "--verificar" in sys.argv:
            print(f"FALTA  {DESTINO}")
            return 1
        baixar()
    obtido = sha256_de(DESTINO)
    if obtido != SHA256:
        print(f"DIVERGE  {DESTINO}\n  esperado {SHA256}\n  obtido   {obtido}")
        print("  o arquivo nao e' o da analise; apague e rode de novo")
        return 1
    linhas = sum(1 for _ in DESTINO.open(encoding="utf-8")) - 1
    print(f"ok  {DESTINO.relative_to(DESTINO.parents[2])}  "
          f"{DESTINO.stat().st_size:,} bytes  {linhas:,} linhas  sha256 {obtido[:12]}...")
    return 0 if (linhas == LINHAS and DESTINO.stat().st_size == BYTES) else 1


if __name__ == "__main__":
    sys.exit(main())
