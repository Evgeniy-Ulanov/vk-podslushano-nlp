# lda.py
# Простая LDA модель для проекта «Подслушано».
# Евгений Уланов

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import tqdm
import pandas as pd


# ----------------------------------------------------------
# Биграммы
# ----------------------------------------------------------

def make_bigrams(texts):
    """
    Создает биграммы на основе списка токенизированных постов.
    """
    bigram = gensim.models.Phrases(texts, min_count=3, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return [bigram_mod[text] for text in texts]


# ----------------------------------------------------------
# Создание словаря и корпуса
# ----------------------------------------------------------

def build_dictionary_and_corpus(texts):
    """
    Создает словарь id2word и корпус для gensim на основе токенов.
    """
    id2word = corpora.Dictionary(texts)

    print(f"Размер словаря до фильтрации: {len(id2word)}")
    id2word.filter_extremes(no_below=3, no_above=0.3)
    print(f"Размер словаря после фильтрации: {len(id2word)}")

    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, corpus


# ----------------------------------------------------------
# Поиск лучшего количества тем
# ----------------------------------------------------------

def compute_coherence(dictionary, corpus, texts, start=5, limit=35, step=5):
    """
    Вычисляет когерентность для разных количеств топиков.
    Возвращает (список моделей, список значений когерентности).
    """

    coherence_values = []
    model_list = []

    for num_topics in tqdm.tqdm(range(start, limit, step)):
        model = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            chunksize=100,
            passes=10,
            alpha=0.1,
            eta=0.1,
        )
        model_list.append(model)

        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(cm.get_coherence())

        print(f"Топиков: {num_topics}, когерентность: {coherence_values[-1]:.4f}")

    return model_list, coherence_values


# ----------------------------------------------------------
# Обучение финальной модели
# ----------------------------------------------------------

def train_final_lda(dictionary, corpus, num_topics=30):
    """
    Обучает финальную модель LDA.
    """

    model = gensim.models.ldamulticore.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=2000,
        passes=20,
        alpha=0.05,
        eta=0.3
    )
    return model


# ----------------------------------------------------------
# Печать топиков
# ----------------------------------------------------------

def print_topics(model, num_topics=30, num_words=10):
    """
    Печатает топики модели LDA (как в ноутбуке).
    """
    topics = model.show_topics(num_topics=num_topics, num_words=num_words, formatted=True)

    for topic in topics:
        print(f"Топик {topic[0] + 1}:")
        formatted = topic[1].replace('*', ': ').replace('+', '\n\t')
        print("\t" + formatted + "\n")


# ----------------------------------------------------------
# Основная функция (для удобства)
# ----------------------------------------------------------

def run_lda_pipeline(df, column='lines_lemmatized', start=5, limit=35, step=5, final_topics=30):
    """
    Выполняет полный цикл:
    1) биграммы
    2) словарь
    3) корпус
    4) выбор количества тем
    5) финальная модель

    Возвращает финальную модель и словарь.
    """

    # 1. исходные токены
    texts = df[column].tolist()

    # 2. биграммы
    texts_bigram = make_bigrams(texts)

    # 3. словарь и корпус
    id2word, corpus = build_dictionary_and_corpus(texts_bigram)

    # 4. оценка количества тем
    print("Идёт подбор количества тем...")
    model_list, coherence_values = compute_coherence(
        dictionary=id2word,
        corpus=corpus,
        texts=texts_bigram,
        start=start,
        limit=limit,
        step=step
    )

    # 5. финальная модель
    print("\nОбучаем финальную модель...")
    final_model = train_final_lda(id2word, corpus, num_topics=final_topics)

    return final_model, id2word, corpus
