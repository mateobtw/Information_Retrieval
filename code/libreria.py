"""
libreria.py — Utilidades para laboratorios de Information Retrieval
====================================================================
Importar desde cualquier notebook del proyecto:

    import sys
    sys.path.append("../../")
    import libreria as ir

Grupos disponibles:
    1. Carga         : load_file, load_corpus
    2. Búsqueda      : search_first, search_all, search_timed, search_bulk
    3. Vocabulario   : get_words, get_corpus_words
"""

import os
import time


# =============================================================================
# GRUPO 1 — Carga
# =============================================================================


def load_file(path: str) -> str:
    """
    Carga un único archivo .txt y retorna su contenido como string.

    Args:
        path: Ruta al archivo .txt

    Returns:
        Contenido del archivo como string.

    Ejemplo:
        text = ir.load_file("../../corpus/11_Alices Adventures in Wonderland.txt")
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_corpus(directory: str) -> dict:
    """
    Carga todos los archivos .txt de un directorio.
    Retorna un diccionario { nombre_archivo: contenido }.

    Args:
        directory: Ruta al directorio que contiene los archivos .txt

    Returns:
        dict con { nombre_archivo (str): contenido (str) }

    Ejemplo:
        corpus = ir.load_corpus("../../corpus/")
    """
    corpus_data = {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                corpus_data[filename] = f.read()

    print(f"Corpus cargado: {len(corpus_data)} archivo(s) desde '{directory}'")
    return corpus_data


# =============================================================================
# GRUPO 2 — Búsqueda lineal
# =============================================================================


def search_first(corpus: dict, query: str) -> str | None:
    """
    Busca un término en el corpus y retorna el PRIMER libro donde aparece.
    Se detiene al encontrar el primer match (más eficiente si solo importa
    saber si existe).

    Args:
        corpus : dict { nombre_archivo: contenido }
        query  : término a buscar (case-sensitive)

    Returns:
        Nombre del primer archivo donde se encontró el término, o None.

    Ejemplo:
        book = ir.search_first(corpus, "Frankenstein")
    """
    for book, content in corpus.items():
        if query in content:
            return book
    return None


def search_all(corpus: dict, query: str) -> list:
    """
    Busca un término en el corpus y retorna TODOS los libros donde aparece.

    Args:
        corpus : dict { nombre_archivo: contenido }
        query  : término a buscar (case-sensitive)

    Returns:
        Lista de nombres de archivos donde se encontró el término.
        Lista vacía si no se encontró en ninguno.

    Ejemplo:
        books = ir.search_all(corpus, "Frankenstein")
    """
    return [book for book, content in corpus.items() if query in content]


def search_timed(corpus: dict, query: str) -> dict:
    """
    Busca un término en el corpus (todos los matches) y mide el tiempo.

    Args:
        corpus : dict { nombre_archivo: contenido }
        query  : término a buscar (case-sensitive)

    Returns:
        dict con:
            - "query"     : término buscado
            - "found_in"  : lista de libros donde apareció
            - "time_s"    : tiempo en segundos
            - "time_ms"   : tiempo en milisegundos

    Ejemplo:
        result = ir.search_timed(corpus, "Alice")
        print(result["time_ms"])
    """
    start = time.time()
    found_in = search_all(corpus, query)
    end = time.time()

    elapsed_s = end - start

    return {
        "query": query,
        "found_in": found_in,
        "time_s": elapsed_s,
        "time_ms": elapsed_s * 1000,
    }


def search_bulk(corpus: dict, queries: set | list) -> dict:
    """
    Busca múltiples términos en el corpus. Mide el tiempo total.

    Args:
        corpus  : dict { nombre_archivo: contenido }
        queries : conjunto o lista de términos a buscar

    Returns:
        dict con:
            - "results"     : { término: [libros donde aparece] }
            - "total_time_s": tiempo total en segundos
            - "total_time_ms: tiempo total en milisegundos
            - "avg_time_ms" : tiempo promedio por término en ms

    Ejemplo:
        words   = ir.get_words(corpus["11_Alices Adventures in Wonderland.txt"])
        bulk    = ir.search_bulk(corpus, words)
        results = bulk["results"]
    """
    results = {}
    start = time.time()

    for query in queries:
        results[query] = search_all(corpus, query)

    end = time.time()
    elapsed_s = end - start

    return {
        "results": results,
        "total_time_s": elapsed_s,
        "total_time_ms": elapsed_s * 1000,
        "avg_time_ms": (elapsed_s * 1000) / len(queries) if queries else 0,
    }


# =============================================================================
# GRUPO 3 — Vocabulario
# =============================================================================


def get_words(text: str) -> set:
    """
    Extrae las palabras únicas de un texto usando split().

    ⚠️ Tokenización básica: no elimina puntuación ni normaliza mayúsculas.
    "Alice", "Alice," y "alice" se tratan como palabras distintas.

    Args:
        text: string con el contenido del texto

    Returns:
        set con las palabras únicas del texto

    Ejemplo:
        words = ir.get_words(corpus["11_Alices Adventures in Wonderland.txt"])
    """
    return set(text.split())


def get_corpus_words(corpus: dict) -> set:
    """
    Extrae las palabras únicas de todo el corpus.

    ⚠️ Tokenización básica: no elimina puntuación ni normaliza mayúsculas.

    Args:
        corpus: dict { nombre_archivo: contenido }

    Returns:
        set con todas las palabras únicas del corpus

    Ejemplo:
        all_words = ir.get_corpus_words(corpus)
        print(len(all_words))
    """
    all_words = set()
    for content in corpus.values():
        all_words.update(content.split())
    return all_words

# =============================================================================
# GRUPO 4 — Índice invertido
# =============================================================================


def build_index(corpus: dict) -> dict:
    """
    Construye un índice invertido a partir del corpus.

    Por cada palabra única de cada documento, registra en qué archivos aparece.

    ⚠️ Tokenización básica: no normaliza ni elimina puntuación.
    "Alice", "Alice," y "alice" generan entradas distintas en el índice.

    Args:
        corpus: dict { nombre_archivo: contenido }

    Returns:
        dict { palabra: [archivos donde aparece] }

    Ejemplo:
        index = ir.build_index(corpus)
    """
    index = {}

    for archivo, contenido in corpus.items():
        vocabulario = set(contenido.split())
        for palabra in vocabulario:
            if palabra in index:
                index[palabra].append(archivo)
            else:
                index[palabra] = [archivo]

    print(f"Índice construido: {len(index)} entradas únicas")
    return index


def search_index(index: dict, query: str) -> list:
    """
    Busca una palabra en el índice invertido. Lookup O(1).

    Args:
        index: dict { palabra: [archivos] }
        query: término a buscar (case-sensitive)

    Returns:
        Lista de archivos donde aparece la palabra.
        Lista vacía si no está en el índice.

    Ejemplo:
        ir.search_index(index, "Alice")
    """
    return index.get(query, [])


def search_index_bulk(index: dict, queries: set | list) -> dict:
    """
    Busca múltiples términos en el índice invertido. Mide el tiempo total.

    Args:
        index  : dict { palabra: [archivos] }
        queries: conjunto o lista de términos a buscar

    Returns:
        dict con:
            - "results"      : { término: [archivos donde aparece] }
            - "total_time_ms": tiempo total en milisegundos
            - "avg_time_ms"  : tiempo promedio por término en ms

    Ejemplo:
        bulk = ir.search_index_bulk(index, words)
    """
    results = {}
    start = time.time()

    for query in queries:
        results[query] = index.get(query, [])

    end = time.time()
    elapsed_ms = (end - start) * 1000

    return {
        "results"      : results,
        "total_time_ms": elapsed_ms,
        "avg_time_ms"  : elapsed_ms / len(queries) if queries else 0,
    }