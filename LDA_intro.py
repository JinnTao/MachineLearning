# -*- coding:utf-8 -*-

from gensim import corpora,models,similarities
from pprint import pprint

if __name__ == "__main__":
    f = open("Data/LDA.txt")
    stop_list = set('for a of the and to in'.split())
    print("After")
    texts = [[word for word in line.strip().lower().split() if word not in stop_list]for line in f]
    print("Text=")
    pprint(texts)

    dictionary = corpora.Dictionary(texts)
    print(dictionary)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    # corpus_tfidf = corpus
    print("TF-IDF")
    for c in corpus_tfidf:
        print(c)

    print("LSI model")
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word= dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print( "LSI Topics")
    pprint(lsi.print_topics(num_topics=2,num_words=5))
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    print("similarity")
    pprint(list(similarity))

    print("LDA Model")
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf,num_topics= num_topics,id2word= dictionary,alpha="auto",eta="auto",
                          minimum_probability=0.001,passes=10)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print("Docunment-Topic:")
    pprint(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print(doc_topic)
    for topic_id in range(num_topics):
        print("topic id:",topic_id)
        pprint(lda.show_topic(topic_id))

    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print("Similarity:")
    pprint(list(similarity))

    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print("USE WITH CARE \n HDA MODEL")
    pprint(topic_result)
    print("HDA Topics")
    print(hda.print_topics(num_topics=2,num_words=5))

