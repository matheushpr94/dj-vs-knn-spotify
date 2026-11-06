"""Roda de Camelot: traduz key + mode do Spotify para a notacao que DJ usa.

DJ nao escolhe a proxima faixa por semelhanca acustica. Escolhe por duas regras
duras: BPM compativel e tonalidade adjacente na roda. Este modulo implementa a
segunda — a primeira e' aritmetica.

A roda tem 24 posicoes: 12 numeros x A (menor) e B (maior). Transicao valida e'
mesma posicao, um numero acima, um abaixo, ou mesmo numero trocando a letra.
"""

# pitch class do Spotify: 0=Do 1=Do# 2=Re 3=Re# 4=Mi 5=Fa 6=Fa# 7=Sol 8=Sol# 9=La 10=La# 11=Si
MAIOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}
MENOR = {9: 8, 4: 9, 11: 10, 6: 11, 1: 12, 8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7}
NOTAS = ["Dó", "Dó#", "Ré", "Ré#", "Mi", "Fá", "Fá#", "Sol", "Sol#", "Lá", "Lá#", "Si"]


def camelot(key: int, mode: int) -> str | None:
    """(key, mode) -> '8B'. Devolve None se a tonalidade nao foi detectada."""
    if key is None or key < 0 or key > 11:
        return None
    return f"{(MAIOR if mode == 1 else MENOR)[int(key)]}{'B' if mode == 1 else 'A'}"


def nome_musical(key: int, mode: int) -> str:
    return f"{NOTAS[int(key)]} {'maior' if mode == 1 else 'menor'}"


def vizinhas(pos: str) -> set[str]:
    """As quatro transicoes que um DJ considera validas a partir de `pos`."""
    n, letra = int(pos[:-1]), pos[-1]
    outra = "A" if letra == "B" else "B"
    return {f"{n}{letra}", f"{n % 12 + 1}{letra}", f"{(n - 2) % 12 + 1}{letra}", f"{n}{outra}"}


def compativel(a: str | None, b: str | None) -> bool:
    return bool(a and b and b in vizinhas(a))


def bpm_ok(t1: float, t2: float, margem: float = .06) -> bool:
    """DJ ajusta o pitch em ate ~6% sem a musica soar estranha."""
    return bool(t1 and t2 and abs(t2 - t1) / t1 <= margem)
