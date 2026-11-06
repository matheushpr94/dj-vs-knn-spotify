"""Roda todos os scripts de analise e regrava `outputs/`.

    uv run python src/rodar_tudo.py            # roda tudo e regrava outputs/
    uv run python src/rodar_tudo.py --conferir # roda e SO' avisa o que mudaria
    uv run python src/rodar_tudo.py p11 p13    # roda so' o que casar com o termo

Resultado que nao reproduz nao e' resultado. `--conferir` responde isso sem
alterar nada: o esperado e' "0 mudou | 0 falha".

Duas guardas:
- saida VAZIA conta como falha. Comparar vazio com vazio daria "igual", e um
  script quebrado em silencio passaria pelo portao;
- o processo filho roda em UTF-8, para o console do Windows nao derrubar um
  script depois de ele ja' ter feito a conta.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
OUTPUTS = RAIZ / "outputs"

# script (sem .py) -> arquivo em outputs/. O numero e' a ordem de leitura:
# primeiro o dado, depois o que ele sustenta, por ultimo o produto.
MAPA = {
    "p01_o_arquivo_e_o_que_promete":       "01-o-arquivo-e-o-que-promete.txt",
    "p02_valores_impossiveis":             "02-valores-impossiveis.txt",
    "p03_popularidade_zero_e_censura":     "03-popularidade-zero-e-censura.txt",
    "p04_o_que_contamina_a_tonalidade":    "04-o-que-contamina-a-tonalidade.txt",
    "p05_o_preparo_pega_defeito":          "05-o-preparo-pega-defeito.txt",
    "selftest_floresta":                   "05b-floresta-em-lotes-da-o-mesmo.txt",
    "p06_o_som_identifica_o_genero":       "06-o-som-identifica-o-genero.txt",
    "p07_da_para_prever_popularidade":     "07-da-para-prever-popularidade.txt",
    "p08_ha_estrutura_alem_do_acaso":      "08-ha-estrutura-alem-do-acaso.txt",
    "p09_filtro_de_popularidade_erra":     "09-filtro-de-popularidade-erra.txt",
    "p10_o_knn_acha_o_que_soa_parecido":   "10-o-knn-acha-o-que-soa-parecido.txt",
    "p11_o_dj_aceitaria_o_knn":            "11-o-dj-aceitaria-o-knn.txt",
    "p12_a_recusa_depende_dos_parametros": "12-a-recusa-depende-dos-parametros.txt",
    "p13_arco_de_energia_ordena_o_set":    "13-arco-de-energia-ordena-o-set.txt",
    "p14_esta_faixa_cabe_na_playlist":     "14-esta-faixa-cabe-na-playlist.txt",
    "p15_pesar_as_features_paga":          "15-pesar-as-features-paga.txt",
    "p16_quantos_grupos_pela_silhueta":    "16-quantos-grupos-pela-silhueta.txt",
    "p17_quantos_grupos_pela_estabilidade": "17-quantos-grupos-pela-estabilidade.txt",
}


def ambiente() -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Reconferir nao e' experimento: sem isto, cada `--conferir` acrescentaria
    # linhas repetidas ao registro de tentativas e ao historico de rodadas.
    env["BANCADA_SILENCIO"] = "1"
    return env


def rodar(nome: str) -> tuple[bool, str]:
    # cwd = raiz do projeto, igual ao `uv run python src/x.py` digitado a mao.
    r = subprocess.run(
        [sys.executable, f"src/{nome}.py"],
        capture_output=True, text=True, encoding="utf-8", env=ambiente(), cwd=str(RAIZ),
    )
    if r.returncode != 0:
        return False, r.stdout + r.stderr
    if not r.stdout.strip():
        return False, "saida vazia: o script terminou sem imprimir nada"
    return True, r.stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("termos", nargs="*", help="roda so' os scripts cujo nome contem o termo")
    p.add_argument("--conferir", action="store_true", help="nao grava; so' diz o que mudaria")
    args = p.parse_args()

    alvos = {k: v for k, v in MAPA.items()
             if not args.termos or any(t in k or t in v for t in args.termos)}
    if not alvos:
        print(f"nenhum script casa com {args.termos}. Disponiveis: {', '.join(MAPA)}")
        return 1

    OUTPUTS.mkdir(exist_ok=True)
    print(f"{'CONFERINDO' if args.conferir else 'RODANDO'} {len(alvos)} script(s)\n")

    mudou, iguais, falhou = [], [], []
    for nome, arquivo in alvos.items():
        ok, saida = rodar(nome)
        if not ok:
            falhou.append(nome)
            print(f"  FALHOU   {nome}")
            print("           " + (saida.strip().splitlines() or ["?"])[-1][:100])
            continue
        destino = OUTPUTS / arquivo
        anterior = destino.read_text(encoding="utf-8") if destino.exists() else None
        igual = anterior is not None and anterior.replace("\r\n", "\n") == saida.replace("\r\n", "\n")
        if not args.conferir:
            destino.write_text(saida, encoding="utf-8", newline="\n")
        (iguais if igual else mudou).append(arquivo)
        print(f"  {'igual   ' if igual else 'MUDOU   '} {nome:<38} -> {arquivo}")

    print(f"\nresumo: {len(iguais)} igual(is) | {len(mudou)} mudou/mudaram | {len(falhou)} falha(s)")
    if mudou and not args.conferir:
        print("        confira com: git diff outputs/")
    return 1 if falhou else 0


if __name__ == "__main__":
    sys.exit(main())
