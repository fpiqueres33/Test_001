from flask import Flask, render_template, request, send_file
from statistics import *
from Generador_resumen import *
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files['file']
    text = file.read().decode('utf-8')

    # Process the text
    doc = process_text(text)

    # Calculate the statistics
    words = count_words(doc)
    lengths, num_sentences, avg_length, median_length = sentence_analysis(doc)

    # Generate histogram
    plot_sentence_length_histogram(lengths)

    # Generate topics using LDA and RAKE
    lemmas = lemmatization(word_tokenization(sentence_segment(text)))
    lda_topics = generate_topics(lemmas, num_topics=3)
    rake_topics = extract_key_phrases(text, num_phrases=3)

    # Perform summarization using similarity matrix
    similarity_matrix = find_similarity(lemmas)
    top_sentences = find_top_sentences(similarity_matrix, sentence_segment(text), percentile=80)

    # Save top_sentences to a text file
    with open('summary.txt', 'w', encoding='utf-8') as file:
        for sentence, _ in top_sentences:
            file.write(sentence + '\n')

    # Perform named entity recognition
    top_entities = named_entity_recognition(text)

    return render_template('result.html', words=words, sentences=num_sentences, avg_length=avg_length,
                           median_length=median_length, lda_topics=lda_topics, rake_topics=rake_topics,
                           top_sentences=top_sentences, top_entities=top_entities)


@app.route('/download/summary.txt')
def download_summary():
    try:
        return send_file('summary.txt', as_attachment=True)
    except FileNotFoundError:
        return 'Error: Summary file not found.'


if __name__ == '__main__':
    app.run()
