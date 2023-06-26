from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import spacy
from gensim.models import LdaModel
from rake_nltk import Rake
from gensim.corpora import Dictionary
from collections import Counter
import logging  # chequear los resultados de la API

nlp = spacy.load('es_core_news_md')
stop_words = set(stopwords.words('spanish'))

def process_text(text):
    """
    Procesa el texto utilizando el modelo de spaCy en español.

    Args:
        text (str): El texto a procesar.

    Returns:
        spacy.tokens.doc.Doc: El documento procesado por spaCy.

    Raises:
        ValueError: Si el texto no es una cadena de caracteres.
    """
    if isinstance(text, str):
        return nlp(text)
    else:
        raise ValueError('El texto debe ser una cadena de caracteres.')

def input_path_and_percentile():
    """
    Solicita al usuario la ruta completa del archivo y el percentil para seleccionar las frases principales.

    Returns:
        tuple: Una tupla que contiene la ruta completa del archivo (formato txt) y el percentil seleccionado.

    Raises:
        ValueError: Si el percentil no está en el rango válido (0-100) y no se puede convertir a entero.
    """
    file_path = input("Indicar archivo con ruta completa (formato txt) --> ")
    percentile_input = input("Introduzca el percentil (0-100) para seleccionar las frases principales (por defecto 80) --> ")
    try:
        percentile = int(percentile_input)
        if percentile < 0 or percentile > 100:
            print("El percentil debe estar entre 0 y 100. Se utilizará el valor por defecto de 80.")
            percentile = 80
    except ValueError:
        print("Entrada no válida. Se utilizará el valor por defecto de 80.")
        percentile = 80
    return file_path, percentile

def read_file(file_path):
    """
    Lee un archivo y devuelve su contenido como una cadena de texto.

    Args:
        file_path (str): La ruta completa del archivo a leer.

    Returns:
        str: El contenido del archivo.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra en la ruta especificada.
        OSError: Si ocurre algún error al abrir o leer el archivo.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().replace('\n', '')
    return data

def sentence_segment(data):
    """
    Segmenta el texto en oraciones utilizando el modelo de spaCy en español.

    Args:
        data (str): El texto a segmentar en oraciones.

    Returns:
        list: Una lista de las oraciones segmentadas.

    """
    nlp = spacy.load('es_core_news_md')  # Selección idioma español tamaño medio
    doc = nlp(data)
    return [sent.text for sent in doc.sents]

def word_tokenization(sentences):
    """
    Tokeniza las oraciones en palabras utilizando NLTK.

    Args:
        sentences (list): Una lista de oraciones a tokenizar.

    Returns:
        list: Una lista de listas de palabras tokenizadas.
    """
    return [word_tokenize(sentence) for sentence in sentences]

def lemmatization(tokens):
    """
    Realiza lematización de las palabras tokenizadas utilizando el modelo de spaCy en español y filtra las palabras de parada y signos de puntuación.

    Args:
        tokens (list): Una lista de listas de palabras tokenizadas.

    Returns:
        list: Una lista de listas de palabras lematizadas.
    """
    lemmas = []
    for token_list in tokens:
        doc = nlp(' '.join(token_list))
        lemmas.append([token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation])
    return lemmas

def find_similarity(lemmas):
    """
    Encuentra la similitud entre las frases utilizando TfidfVectorizer y cosine_similarity de scikit-learn.

    Args:
        lemmas (list): Una lista de listas de palabras lematizadas.

    Returns:
        numpy.ndarray: Una matriz de similitud de coseno entre las frases.
    """
    sentences = [' '.join(lemma) for lemma in lemmas]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(X, X)
    return similarity_matrix

def generate_topics(lemmas, num_topics=3):
    """
    Genera los tópicos utilizando LDA (Latent Dirichlet Allocation) y RAKE (Rapid Automatic Keyword Extraction).

    Args:
        lemmas (list): Una lista de listas de palabras lematizadas.
        num_topics (int): El número de tópicos a generar (por defecto 3).

    Returns:
        list: Una lista de tuplas que representan los tópicos generados.
    """
    dictionary = Dictionary(lemmas)
    corpus = [dictionary.doc2bow(lemma) for lemma in lemmas]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    topics = lda_model.print_topics()
    return topics

def extract_key_phrases(text, num_phrases=3):
    """
    Extrae las frases clave del texto utilizando RAKE (Rapid Automatic Keyword Extraction).

    Args:
        text (str): El texto del cual extraer las frases clave.
        num_phrases (int): El número de frases clave a extraer (por defecto 3).

    Returns:
        list: Una lista de tuplas que contienen el puntaje de relevancia normalizado y la frase clave extraída.
    """
    rake = Rake()
    rake.extract_keywords_from_text(text)
    key_phrases_with_scores = rake.get_ranked_phrases_with_scores()[:num_phrases]
    max_score = max(score for score, _ in key_phrases_with_scores)
    key_phrases_with_scores_normalized = [(score / max_score, phrase) for score, phrase in key_phrases_with_scores]
    return key_phrases_with_scores_normalized

def find_top_sentences(similarity_matrix, sentences, percentile=80):
    """
    Encuentra las oraciones principales basadas en la matriz de similitud y un percentil dado.

    Args:
        similarity_matrix (numpy.ndarray): La matriz de similitud de coseno entre las frases.
        sentences (list): Una lista de las oraciones originales.
        percentile (int): El percentil para seleccionar las oraciones principales (por defecto 80).

    Returns:
        list: Una lista de tuplas que contienen las oraciones principales y sus puntajes de similitud.
    """
    sum_similarities = np.sum(similarity_matrix, axis=1)

    # Normaliza metrics
    total_sum = np.sum(sum_similarities)
    if total_sum != 0:
        sum_similarities = sum_similarities / total_sum

    # Calcula el  threshold score a un determinado percentil (para poder ajustar la longitud de la respuesta)
    threshold_score = np.percentile(sum_similarities, percentile)

    # Obtiene el index de las oraciones con mayor similarity scores higher y mayor que el threshold score
    top_indexes = [index for index in range(len(sentences)) if sum_similarities[index] > threshold_score]

    # Ordena las frases en base a sus similarity scores
    top_sentences = sorted([(sentences[index], sum_similarities[index]) for index in top_indexes],
                           key=lambda x: x[1], reverse=True)

    return top_sentences

def named_entity_recognition(text):
    """
    Reconoce las entidades nombradas en el texto utilizando el modelo de spaCy en español.

    Args:
        text (str): El texto en el cual reconocer las entidades nombradas.

    Returns:
        list: Una lista de tuplas que contienen las entidades nombradas y sus etiquetas.
    """
    doc = nlp(text)
    named_entities = []

    for ent in doc.ents:
        if ent.label_ != 'MISC':
            named_entities.append((ent.text.upper(), ent.label_))
    entity_counts = Counter(named_entities)

    # Muestra el top 10 de entidades.
    top_entities = entity_counts.most_common(10)

    return top_entities
