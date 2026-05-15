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
    4. Índice        : build_index, search_index, search_index_bulk
    5. Modelo vect.  : tokenize, build_vocabulary, tf, tf_norm, idf,
                       tf_docs, df_docs
    6. BM25          : idf_bm25, bm25_score, bm25_rank
    7. TF-IDF sklearn: preprocess, cosine_sim, tfidf_buscar
"""

import os
import time
import math
import re
import numpy as np
from collections import Counter
from nltk.stem import SnowballStemmer

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
        "results": results,
        "total_time_ms": elapsed_ms,
        "avg_time_ms": elapsed_ms / len(queries) if queries else 0,
    }


# =============================================================================
# GRUPO 5 — Modelo vectorial
# =============================================================================


def tokenize(text: str) -> list:
    """
    Convierte un texto en lista de tokens normalizados.

    Aplica: minúsculas, eliminación de puntos, comas y BOM (\\ufeff).

    ⚠️ Mejora sobre get_words(): normaliza mayúsculas y elimina puntuación básica,
    por lo que "Alice", "alice" y "alice," se tratan como el mismo token.

    Args:
        text: string con el contenido del texto

    Returns:
        list con los tokens normalizados del texto

    Ejemplo:
        tokens = ir.tokenize("Las Islas Galápagos, únicas.")
        # ['las', 'islas', 'galápagos', 'únicas']
    """
    return text.lower().replace(".", "").replace(",", "").replace("\ufeff", "").split()


def build_vocabulary(docs_tokenizados: list) -> list:
    """
    Construye el vocabulario ordenado de un corpus tokenizado.

    Extrae todos los términos únicos de la lista de documentos tokenizados
    y los retorna ordenados alfabéticamente. El orden fijo garantiza que
    cada término corresponda siempre al mismo índice de columna en la matriz.

    Args:
        docs_tokenizados: list de list — cada sublista es un documento tokenizado

    Returns:
        list ordenada con todos los términos únicos del corpus

    Ejemplo:
        docs  = [ir.tokenize(doc) for doc in documentos]
        vocab = ir.build_vocabulary(docs)
        print(len(vocab))
    """
    vocabulario = set()
    for doc in docs_tokenizados:
        vocabulario.update(doc)
    return sorted(vocabulario)


def tf(termino: str, doc: list) -> int:
    """
    Calcula la Term Frequency cruda de un término en un documento.

    TF(t, d) = conteo de t en d

    Args:
        termino : término a calcular
        doc     : documento como lista de tokens

    Returns:
        int — número de veces que aparece el término

    Ejemplo:
        tokens = ir.tokenize("el gato y el perro")
        ir.tf("el", tokens)  # 2
    """
    return sum(token == termino for token in doc)


def tf_norm(termino: str, doc: list) -> float:
    """
    Calcula la Term Frequency normalizada de un término en un documento.

    TF(t, d) = conteo de t en d / total de tokens en d

    La normalización evita que documentos más largos tengan ventaja
    por tener conteos absolutos más altos.

    Args:
        termino : término a calcular
        doc     : documento como lista de tokens

    Returns:
        float entre 0 y 1

    Ejemplo:
        tokens = ir.tokenize("el gato y el perro")
        ir.tf_norm("el", tokens)  # 0.4
    """
    return sum(token == termino for token in doc) / len(doc)


def idf(termino: str, docs: list) -> float:
    """
    Calcula el Inverse Document Frequency de un término sobre el corpus.

    IDF(t) = log(N / df(t))

    donde df(t) es el número de documentos que contienen el término t.
    El logaritmo suaviza el crecimiento para que términos muy raros
    no dominen desproporcionadamente el peso final.

    Args:
        termino : término a calcular
        docs    : lista de documentos tokenizados

    Returns:
        float — mayor valor indica término más raro (más discriminativo)

    Ejemplo:
        docs = [ir.tokenize(doc) for doc in documentos]
        ir.idf("galápagos", docs)
    """
    df = sum(termino in doc for doc in docs)
    return math.log(len(docs) / df) if df > 0 else 0.0


def tf_docs(docs_tokenizados: list) -> list:
    """
    Construye el TF crudo para cada documento del corpus usando Counter.

    A diferencia de tf() y tf_norm() (que operan término a término),
    esta función procesa cada documento en un solo recorrido — mucho más
    eficiente a escala de corpus completo.

    El TF aquí es crudo (conteo absoluto, sin normalizar por longitud),
    lo que lo hace compatible con BM25, que maneja la normalización
    internamente con el parámetro b.

    Args:
        docs_tokenizados: list de list — cada sublista es un documento tokenizado

    Returns:
        list de Counter — un Counter por documento con { término: conteo_crudo }

    Ejemplo:
        docs    = [ir.tokenize(doc) for doc in corpus.values()]
        tf_list = ir.tf_docs(docs)
        tf_list[0]["the"]  # conteo de "the" en el primer documento
    """
    return [Counter(doc) for doc in docs_tokenizados]


def df_docs(tf_docs_list: list) -> Counter:
    """
    Calcula la Document Frequency de cada término en el corpus.

    DF(t) = número de documentos que contienen el término t.

    Itera sobre los Counters de TF y acumula con Counter.update(),
    que incrementa en 1 por cada término presente en cada documento.

    Args:
        tf_docs_list: list de Counter — salida de tf_docs()

    Returns:
        Counter { término: df } con la frecuencia de documentos de cada término

    Ejemplo:
        docs    = [ir.tokenize(doc) for doc in corpus.values()]
        tf_list = ir.tf_docs(docs)
        df      = ir.df_docs(tf_list)
        df["the"]   # en cuántos documentos aparece "the"
    """
    df = Counter()
    for tf_doc in tf_docs_list:
        df.update(tf_doc.keys())
    return df


# =============================================================================
# GRUPO 6 — BM25
# =============================================================================


def idf_bm25(df_dict: Counter, N: int) -> dict:
    """
    Calcula el IDF con la fórmula de BM25 (Robertson et al.) para todo el corpus.

    IDF_BM25(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

    A diferencia del IDF clásico log(N/df), esta fórmula:
    - Nunca produce valores negativos (el +1 externo lo garantiza)
    - Es más suave con términos muy comunes
    - Penaliza menos drásticamente los términos que aparecen en todos los docs

    Args:
        df_dict : Counter { término: df } — salida de df_docs()
        N       : número total de documentos en el corpus

    Returns:
        dict { término: idf_value }

    Ejemplo:
        docs    = [ir.tokenize(doc) for doc in corpus.values()]
        tf_list = ir.tf_docs(docs)
        df      = ir.df_docs(tf_list)
        idf     = ir.idf_bm25(df, N=len(docs))
    """
    return {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1) for term, df in df_dict.items()
    }


def bm25_score(
    tf_doc: Counter,
    doc_length: int,
    query_terms: list,
    idf_dict: dict,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    Calcula el score BM25 de un documento dado un query.

    BM25(d, q) = Σ IDF(t) · [TF(t,d) · (k1+1)] / [TF(t,d) + k1·(1 - b + b·|d|/avgdl)]

    Parámetros:
        k1 — controla la saturación del TF. A mayor k1, más lento satura.
             Con k1=0 el TF no influye; con k1→∞ se comporta como TF puro.
        b  — peso de la normalización por longitud.
             b=0: sin normalización. b=1: normalización completa.

    Args:
        tf_doc      : Counter { término: conteo_crudo } del documento
        doc_length  : número de tokens del documento
        query_terms : lista de tokens del query (salida de tokenize())
        idf_dict    : dict { término: idf } — salida de idf_bm25()
        avgdl       : longitud promedio de los documentos del corpus
        k1          : parámetro de saturación (default 1.2)
        b           : parámetro de normalización por longitud (default 0.75)

    Returns:
        float — score BM25 del documento (mayor = más relevante)

    Ejemplo:
        score = ir.bm25_score(tf_list[0], doc_lengths[0], ir.tokenize(query), idf, avgdl)
    """
    score = 0.0
    for term in query_terms:
        if term not in idf_dict:
            continue  # término fuera del vocabulario → aporte nulo
        tf_term = tf_doc.get(term, 0)
        score += (
            idf_dict[term]
            * (tf_term * (k1 + 1))
            / (tf_term + k1 * (1 - b + b * (doc_length / avgdl)))
        )
    return score


def bm25_rank(
    consulta: str,
    titulos: list,
    tf_docs_list: list,
    doc_lengths: list,
    idf_dict: dict,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
    top_k: int = 10,
) -> list:
    """
    Rankea todos los documentos del corpus por score BM25 dado un query.

    Args:
        consulta      : string con el query de búsqueda
        titulos       : list de nombres de documento (mismo orden que tf_docs_list)
        tf_docs_list  : list de Counter — salida de tf_docs()
        doc_lengths   : list de int — longitud en tokens de cada documento
        idf_dict      : dict { término: idf } — salida de idf_bm25()
        avgdl         : longitud promedio de los documentos
        k1            : parámetro de saturación (default 1.2)
        b             : parámetro de normalización por longitud (default 0.75)
        top_k         : número de documentos a retornar (default 10)

    Returns:
        list de dict [{ "título": str, "score": float }] ordenada por score descendente

    Ejemplo:
        docs        = [ir.tokenize(doc) for doc in corpus.values()]
        titulos     = list(corpus.keys())
        tf_list     = ir.tf_docs(docs)
        df          = ir.df_docs(tf_list)
        idf         = ir.idf_bm25(df, N=len(docs))
        doc_lengths = [len(doc) for doc in docs]
        avgdl       = sum(doc_lengths) / len(doc_lengths)

        ranking = ir.bm25_rank("mystery and detective", titulos, tf_list,
                                doc_lengths, idf, avgdl)
    """
    query_terms = tokenize(consulta)
    scores = [
        bm25_score(tf_docs_list[i], doc_lengths[i], query_terms, idf_dict, avgdl, k1, b)
        for i in range(len(titulos))
    ]

    # Ordenar por score descendente y tomar top_k
    ranking = sorted(
        [{"título": titulos[i], "score": scores[i]} for i in range(len(titulos))],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranking[:top_k]


# =============================================================================
# GRUPO 7 — TF-IDF (sklearn)
# =============================================================================
#
# Estas funciones complementan el pipeline de sklearn para corpus a escala
# real. El flujo típico es:
#
#   docs_proc    = [ir.preprocess(doc) for doc in corpus.values()]
#   vectorizer   = TfidfVectorizer(stop_words='english')
#   tfidf_matrix = vectorizer.fit_transform(docs_proc)
#   doc_ids      = list(corpus.keys())
#
#   resultados = ir.tfidf_buscar("query aquí", vectorizer, tfidf_matrix, doc_ids)
#
# Nota sobre sparse matrices: la matriz TF-IDF de un corpus grande es dispersa
# (la mayoría de documentos no contienen la mayoría de términos). NUNCA llamar
# .toarray() sobre la matriz completa — puede requerir varios GB de RAM.
# Las funciones de este grupo operan en sparse de forma segura.

_stemmer = SnowballStemmer("english")


def preprocess(texto: str) -> str:
    """
    Preprocesa un texto para el pipeline TF-IDF de sklearn.

    Aplica tres pasos en orden:
        1. Minúsculas
        2. Elimina todo carácter que no sea letra o espacio (puntuación, números,
           caracteres especiales, BOM)
        3. Stemming con SnowballStemmer (inglés) — reduce cada token a su raíz

    Los stopwords NO se eliminan aquí: se delegan al TfidfVectorizer con
    stop_words='english', que los descarta al calcular los pesos TF-IDF.

    Diferencia con tokenize(): tokenize() solo elimina puntos y comas y retorna
    una lista. preprocess() hace limpieza completa, aplica stemming y retorna
    un string listo para ser consumido por TfidfVectorizer.

    Args:
        texto : str — texto crudo del documento o de una consulta.

    Returns:
        str — tokens stemmeados separados por espacios.

    Ejemplo:
        ir.preprocess("The quick brown foxes are running")
        # 'the quick brown fox are run'

        # Uso típico sobre un corpus completo:
        docs_proc = [ir.preprocess(doc) for doc in corpus.values()]
    """
    texto = texto.lower()
    texto = re.sub(r"[^a-z\s]", "", texto)
    tokens = texto.split()
    return " ".join(_stemmer.stem(t) for t in tokens)


def cosine_sim(query_vec, doc_matrix) -> np.ndarray:
    """
    Calcula la similitud del coseno entre una query y todos los documentos.

    Implementación manual sobre matrices dispersas (sparse-safe): no convierte
    la matriz de documentos a densa en ningún momento, por lo que es segura
    para corpus grandes.

    Fórmula:
        sim(q, d) = (q · d) / (||q|| · ||d||)

    Comparación con sklearn.metrics.pairwise.cosine_similarity:
        - Esta implementación produce resultados numéricamente equivalentes
          (diferencia < 1e-15 por redondeo de punto flotante).
        - sklearn es más rápida en la práctica porque usa rutinas BLAS/LAPACK
          optimizadas internamente.
        - Esta versión es útil para entender el cálculo paso a paso y para
          contextos donde no se puede o no se quiere depender de sklearn.

    Args:
        query_vec  : sparse matrix (1, n_terms) — vector de la query vectorizada.
                     Salida de vectorizer.transform([query_procesada]).
        doc_matrix : sparse matrix (n_docs, n_terms) — matriz TF-IDF del corpus.
                     Salida de vectorizer.fit_transform(docs).

    Returns:
        np.ndarray de shape (n_docs,) con la similitud de cada documento
        respecto a la query. Valores en [0, 1]; mayor = más relevante.

    Ejemplo:
        query_vec = vectorizer.transform([ir.preprocess("beach")])
        sims      = ir.cosine_sim(query_vec, tfidf_matrix)
        top5      = sims.argsort()[::-1][:5]
    """
    # Convertir la query a array denso — es solo 1 × n_terms, entra en RAM
    q = np.asarray(query_vec.todense()).flatten()

    # Producto punto: doc_matrix @ q sin convertir doc_matrix a densa
    # doc_matrix es sparse (n_docs, n_terms); q es denso (n_terms,)
    # El resultado es denso (n_docs,)
    dots = doc_matrix.dot(q)

    # Norma de la query
    norm_q = np.linalg.norm(q)

    # Norma de cada documento usando operaciones sparse
    # doc_matrix.multiply(doc_matrix) eleva al cuadrado elemento a elemento
    # .sum(axis=1) suma por fila → (n_docs, 1); aplanar a (n_docs,)
    norm_docs = np.sqrt(
        np.asarray(doc_matrix.multiply(doc_matrix).sum(axis=1)).flatten()
    )

    # Denominador: evitar división por cero si algún doc o la query son cero
    denominador = norm_docs * norm_q
    denominador[denominador == 0] = 1.0

    return dots / denominador


def tfidf_buscar(
    query: str,
    vectorizer,
    tfidf_matrix,
    doc_ids: list,
    top_n: int = 10,
) -> list:
    """
    Busca los documentos más relevantes para una consulta usando TF-IDF + coseno.

    Pipeline completo:
        1. Preprocesa la query con preprocess() — mismo pipeline que los documentos
        2. Vectoriza la query con el vectorizer ya ajustado (.transform, no .fit)
        3. Calcula la similitud del coseno con cosine_sim()
        4. Retorna los top_n documentos ordenados por similitud descendente

    Args:
        query        : str    — consulta en texto libre (inglés).
        vectorizer   : TfidfVectorizer ya ajustado con fit_transform sobre el corpus.
        tfidf_matrix : sparse matrix (n_docs, n_terms) — salida de fit_transform.
        doc_ids      : list   — nombres de los documentos, mismo orden que tfidf_matrix.
        top_n        : int    — número de resultados a retornar (default 10).

    Returns:
        list de dict [{ "doc_id": str, "similitud": float }]
        ordenada por similitud descendente. Solo incluye documentos con sim > 0.

    Ejemplo:
        # Preparar el corpus
        docs_proc    = [ir.preprocess(doc) for doc in corpus.values()]
        doc_ids      = list(corpus.keys())
        vectorizer   = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(docs_proc)

        # Buscar
        resultados = ir.tfidf_buscar("sea adventure", vectorizer, tfidf_matrix, doc_ids)
        for r in resultados:
            print(r["doc_id"], r["similitud"])
    """
    query_proc = preprocess(query)
    query_vec = vectorizer.transform([query_proc])

    sims = cosine_sim(query_vec, tfidf_matrix)
    idx_top = sims.argsort()[::-1][:top_n]

    return [
        {"doc_id": doc_ids[i], "similitud": round(float(sims[i]), 4)}
        for i in idx_top
        if sims[i] > 0
    ]
