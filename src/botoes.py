"""Botoes de linha de comando: mexer em parametro sem editar arquivo.

POR QUE ISTO EXISTE. Experimentar mudando a constante dentro do `.py` funciona e
tem dois custos: suja o git (e ai' `--conferir` acusa mudanca que voce nem queria
guardar) e e' facil esquecer o valor antigo. Pela linha de comando, o experimento
nao deixa rastro — fechou o terminal, acabou.

USO no script:

    from botoes import botao
    MARGEM = botao("--margem", 0.06, float, "quanto o BPM pode diferir")
    N_SET  = botao("--set", 20, int, "quantas faixas no set")

E ai':

    uv run python src/p13_arco_de_energia_ordena_o_set.py --margem 0.10 --set 30
    uv run python src/p13_arco_de_energia_ordena_o_set.py --ajuda        # lista os botoes

REGRA DE OURO SOBRE `--semente`: nao mexa nela para "melhorar" um numero. Se um
resultado muda muito ao trocar a semente, ele era sorte. Trocar semente e' TESTE
DE ROBUSTEZ, nao busca de resultado melhor.
"""
import sys

_REGISTRO: list[tuple[str, object, str]] = []


def botao(flag: str, padrao, tipo=str, ajuda: str = ""):
    """Le `flag` de sys.argv, ou devolve `padrao`. Registra para o `--ajuda`.

    O registro DEDUPLICA por flag, e a razao e' um efeito colateral de import que
    apareceu no primeiro teste: `p15_pesar_as_features_paga.py` importa de `p14_esta_faixa_cabe_na_playlist.py`, e importar
    executa os `botao()` do outro arquivo. Sem deduplicar, o `--ajuda` do pesos
    listava os botoes do pertence junto. O valor lido continua o mesmo nos dois —
    e' a mesma flag — entao deduplicar so' limpa a exibicao.
    """
    if flag not in {f for f, _, _ in _REGISTRO}:
        _REGISTRO.append((flag, padrao, ajuda))
    if flag in sys.argv:
        i = sys.argv.index(flag) + 1
        if i >= len(sys.argv):
            sys.exit(f"erro: {flag} precisa de um valor. Ex: {flag} {padrao}")
        try:
            return tipo(sys.argv[i])
        except ValueError:
            sys.exit(f"erro: {flag} espera {tipo.__name__}, veio '{sys.argv[i]}'")
    return padrao


def mudou() -> list[str]:
    """Quais botoes sairam do padrao — para o script avisar que nao e' a rodada oficial."""
    return [f for f, _, _ in _REGISTRO if f in sys.argv]


def ajuda_se_pedida(descricao: str = "") -> None:
    """Se veio `--ajuda`, imprime os botoes e sai. Chamar no fim do main()."""
    if "--ajuda" not in sys.argv and "-h" not in sys.argv:
        return
    print(f"\n{descricao}\n" if descricao else "")
    print(f"  {'botao':<16} {'padrao':>10}   o que muda")
    print("  " + "-" * 68)
    for flag, padrao, txt in _REGISTRO:
        print(f"  {flag:<16} {str(padrao):>10}   {txt}")
    print("\n  os padroes sao a rodada oficial — mudar um botao NAO deve ser salvo")
    print("  por cima do arquivo em outputs/.\n")
    sys.exit(0)


def aviso_se_mudou() -> None:
    """Imprime aviso quando o script nao esta' na configuracao oficial."""
    m = mudou()
    if m:
        print(f"  [experimento: {' '.join(m)} — NAO e' a rodada oficial, "
              f"nao salve por cima de outputs/]\n")
