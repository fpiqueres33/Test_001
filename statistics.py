import spacy
import numpy as np
import locale
import matplotlib.pyplot as plt

locale.setlocale(locale.LC_ALL, 'es_ES')

nlp = spacy.load('es_core_news_md')

def process_text(text):
    """
    Procesa el texto utilizando el modelo de spaCy.

    Args:
        text (str): El texto a procesar.

    Returns:
        spacy.tokens.doc.Doc: El texto procesado por spaCy.

    Raises:
        ValueError: Si el texto no es una cadena de caracteres.
    """
    if isinstance(text, str):
        return nlp(text)
    else:
        raise ValueError('El texto debe ser una cadena de caracteres.')

def count_words(doc):
    """
    Cuenta la cantidad de palabras en el documento.

    Args:
        doc (spacy.tokens.doc.Doc): El documento procesado.

    Returns:
        str: La cantidad de palabras formateada.
    """
    words = [token.text for token in doc if not token.is_space]
    return locale.format_string('%d', len(words), grouping=True)

def sentence_analysis(doc):
    """
    Realiza un análisis de las oraciones en el documento.

    Args:
        doc (spacy.tokens.doc.Doc): El documento procesado.

    Returns:
        tuple: Una tupla que contiene las longitudes de las oraciones, la cantidad de oraciones formateada,
               la longitud promedio de las oraciones formateada y la longitud mediana de las oraciones formateada.
    """
    lengths = [len(sent.text.split()) for sent in doc.sents]
    return lengths, locale.format_string('%d', len(lengths), grouping=True), \
           locale.format_string('%.2f', np.mean(lengths) if lengths else 0, grouping=True), \
           locale.format_string('%.2f', np.median(lengths) if lengths else 0, grouping=True)

def plot_sentence_length_histogram(lengths):
    """
    Genera un histograma de las longitudes de las oraciones.

    Args:
        lengths (list): Una lista de las longitudes de las oraciones.

    Returns:
        None
    """
    plt.figure(figsize=(4, 2))
    plt.hist(lengths, bins=min(10, len(lengths)))
    plt.title('Histograma de Longitudes de Oraciones')
    plt.xlabel('Longitud de Oración')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.savefig('static/sentence_histogram.png')


