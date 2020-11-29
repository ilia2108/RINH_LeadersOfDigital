#! /usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import zipfile
import wget
import os
from nltk.corpus import stopwords
from summa.summarizer import summarize
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import nltk
import pymorphy2
import gensim
import requests
from rutermextract import TermExtractor

# ТЕГ - это объект, которым можно охарактеризовать проект или пользователя. Напр: молекулярная физика - тег проекта или навык пользователя)
# Инициализация модели для ключевых фраз из предложений
term_extractor = TermExtractor()

# Скачивание базы слов, которые необходимо удалить из ключевых (местоимения, предлоги и т.д.)
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
# Дополняем базу собственными словами (В дальнейшем файл можно менять в зависимости от успеха работы модуля get_key_phrases)
pronouns = ' '.join(open('stopwordsadd.txt', 'r').read().split('\n')).split(' ,')
russian_stopwords += pronouns
# Инициализация модели для приведения слов к лемме (ее нормальной форме)
morph = pymorphy2.MorphAnalyzer()

# Устанавливаем компоненты сервера
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Скачивание и установка модели для поиска семантической связи между словами. (Для сравнение двух тегов, напр: ФИЗИКА/ИСТОРИЯ)
model_url = 'http://vectors.nlpl.eu/repository/11/180.zip'  # Множество вариантов моделей для использования, можно тестировать на необходимый запас слов
model_file = model_url.split('/')[-1]

if not (os.path.isfile(model_file)):
    m = wget.download(model_url)

with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

# Установка модели для анализа рецензий на научные работы
tokenizer = RegexTokenizer()

model_url = 'https://storage.b-labs.pro/models/fasttext-social-network-model.bin'
model_file = model_url.split('/')[-1]

if not (os.path.isfile(model_file)):
    m = wget.download(model_url)

tokenizer = RegexTokenizer()
FastTextSocialNetworkModel.MODEL_PATH = 'fasttext-social-network-model.bin'
model_sent = FastTextSocialNetworkModel(tokenizer=tokenizer)

"""
Параметры, вияющие на рейтинг в рекомендуемом пользователю листе. 
Могут быть настроены как вручную, 
так и с использованием адаптивных алгоритмов 
в будущем для наиболее успешной рекомендации.

tags_correct_const - Количество необходимых совпадений по тегам пользователя и проекта для дальнейшего анализа статей пользователя на узкоспециализированные теги
semantic_const - Мера семантической близости (параметр критерия схожести слов). Чем выше параметр, тем выше порог для отброра схожих слов. (Напр: 'онкологический','онкология' = 0.6, 'дерево','автомобиль' = 0.1) 
ratio_const - Обозначает степень суммаризации текста. Чем ниже парамтер, тем больше сжатие
additional_weight_keys_annotation - Вес влияния тегов, полученных моделью с аннотаций. Чем выше вес, тем большую роль играет аннотация
additional_weight_project_tag -  влияния тегов, полученных моделью с присвоенных пользователем тегов проекту. Чем выше вес, тем большую роль играют теги проекта 
count_projects_weight - Вес влияния тегов, полученных моделью с присвоенных пользователем тегов к проекту. Чем выше вес, тем большую роль играют теги проекта 
count_articles_weight - Вес влияния количества статей на оценку кандидата моделью. Чем выше вес, тем большую роль играет количество статей
candidate_rating_weight - Вес влияния оценки профиля кандидата другими пользователями. Чем выше вес, тем большую роль играет оценка
citation_index_weight - Вес влияния среднего количества цитируемости статей кандидата. Чем выше вес, тем большую роль играет цитируемость  
pos_const - Вес влияния позитивных рецензий на статьи кандидата. Чем выше вес, тем большую роль играют позитивные рецензии
neg_const - Вес влияния негативных рецензий на статьи кандидата. Чем выше вес, тем большую роль играют негативные рецензии
h_index_weight - Вес влияния Индекса Хирша на оценку кандидата
"""

tags_correct_const = 6
semantic_const = 0.4
ratio_const = 0.1
additional_weight_keys_annotation = 0.4
additional_weight_project_tag = 2.0
count_projects_weight = 1.0
count_articles_weight = 1.0
candidate_rating_weight = 0.2
citation_index_weight = 1.0
pos_const = 1.0
neg_const = 1.0
h_index_weight = 1.0


@app.route("/get_candidates", methods=['POST'])
@cross_origin()
def get_candidates():
    """
    get_candidates(request.post) - вход: json, выход: json
    Создание листа кандидатов на проект

    Входной JSON:
    projects_tags - теги, присвоенные проекту
    annotation - аннотация, написанная для проекта
    candidates - Кандидаты для приглашения
      - ID (Уникальный идентификационный номер кандидата)
      - tags (Теги кандидата, введеные при регистрации)
        - name (Название тега)
        - stage (Уровень навыка)
      - articles (Статьи кандидата)
        - reviews (Рецензии на статьи)
      - rating (Оценки, полученные кандидатом от других пользователей)
      - projects (Количество проектов пользователя)

    Выходной Json:
    Лист кандидатов, отсортированный по убыванию оценки модели.
    Вид: 'result':[(ID,оценка)]
    """
    json = request.get_json()
    project_tags = json['project_tags']
    annotation = json['annotation']
    candidates = json['candidates']

    keys_annotation_tags = get_key_phrases(annotation)

    keys_annotation_tags = lemmatization(keys_annotation_tags)
    project_tags = lemmatization(project_tags)

    short_list = {}

    # Подбор кандидатов
    for candidate in candidates:
        # Признаки кандидата
        candidate_ID = candidate['ID']
        candidate_tags = candidate['tags']
        articles = candidate['articles']
        count_articles = len(articles)
        candidate_rating = candidate['rating']
        count_projects = len(candidate['projects'])
        citation = candidate['citation']
        h_index = candidate['h_index']
        # Начальная оценка кандидата
        candidate_score = max(count_projects, 0.000001) * count_projects_weight + max(candidate_rating,
                                                                                      0.000001) * candidate_rating_weight + max(
            count_articles, 0.000001) / count_articles_weight + max(citation, 0.000001) * citation_index_weight + max(h_index, 0.000001) * h_index 
        tags_correct = 0
        keys_annotation_tags, project_tags = completion(keys_annotation_tags, project_tags)

        for keys_annotation_tag, project_tag in zip(keys_annotation_tags, project_tags):
            for candidate_tag in candidate_tags:
                # Корректировка оценки, на основе обнаруженных тегов
                x, y = change_candidate_score(keys_annotation_tag, project_tag, candidate_tag)
                candidate_score += x;
                tags_correct += y
            # Если теги были найдены, то
            # 1) Начинаем более узкоспециализированный поиск внутри статей для дополнительной оценки кандидата
            # 2) Анализ тональности отзывов для дополнительной оценки кандидата
            if tags_correct > tags_correct_const:
                for article in articles:
#                     _=antiplag(article)
                    key_phrases_article = get_key_phrases(article)
                    for key_phrase_article in key_phrases_article:
                        x, y = change_candidate_score(key_phrase_article, False, candidate_tag)
                        candidate_score += x;
                        tags_correct += y
                    reviews = arcticle['reviews']
                    for review in reviews:
                        pos, neg = sentiments(review)
                        candidate_score += pos * pos_const - neg * neg_const

        # Добавляем кандидата в список
        short_list[candidate_ID] = candidate_score

    return jsonify({'result': sorted(short_list.items(), key=lambda x: (x[1], x[0]))[::-1]})


def get_proximity(tag_1, tag_2):
    """
    get_proximity(tag_1,tag_2) - вход: str, выход: int/str

    Использование модели NLP rusvectores, обученную на Национальном корпусе русского языка (НКРЯ)
    для поиска семантической близости между словами.
    Напр: 'онкологический','онкология' = 0.6, 'дерево','автомобиль' = 0.1

    Входной str:
    tag_1, tag_2 - слова, которые сравниваются

    Выходной int/str:
    int: Мера семантической близости
    str: Отсутствие слов в NLP модели
    """
    tag_1, tag_2, stage = tags_in_model(tag_1, tag_2)
    if stage:
        return model.similarity(tag_1, tag_2)
    else:
        return 'Not Found'


def get_summary(text):
    """
    get_summary(text) - вход: str, выход: str
  
    Суммаризация (автореферирование/сжатие > 10 раз) текста для:
    1) Ускоренной обработки информации другими моделями
    2) Предоставление сжатых статей кандидатов для ознакомления
    пользователям
  
    Входной str:
    text - текст статьи
  
    Выходной str:
    text - суммаризированный текст
    """
    return summarize(text, ratio=ratio_const)


def get_key_phrases(text):
    """
    get_key_phrases(text) - вход: str, выход: list (str)
  
    Получение ключевых фраз из текста (в том числе выделение тегов)
    для последующей обработки
  
    Входной str:
    text - определенный текст
  
    Выходной list (str):
    key_phrases - лист ключевых слов
    """
    text = get_summary(text)
    x = text.split('\n')
    x = [i for i in ' '.join(x).split()]
    x = [term.normalized for term in term_extractor(' '.join(x))]
    key_phrases = []
    for i in x:
        if morph.parse(i)[0][2] not in russian_stopwords:
            key_phrases.append(morph.parse(i)[0][2])
    return key_phrases


def tags_in_model(tag_1, tag_2):
    """
    tags_in_model(tag_1,tag_2) - вход: str, выход: str/boolean

    Поиск слов в базе данных модели семантической близости и 
    их редактирование для правильной работы модели.

    Входной str:
    tag_1, tag_2 - слова, которые предобрабатывают

    Выходной str/boolean:
    str: предобработанные слова
    boolean: True - слова нашлись в базе данных модели
    False - слова не нашлись в базе данных модели
    """
    st1, st2 = 0, 0
    for i in ['_VERB', '_ADJ', '_NOUN']:
        if tag_1 + i in model and not (st1):
            st1 = 1
            tag_1 = tag_1 + i
        if tag_2 + i in model and not (st2):
            st2 = 1
            tag_2 = tag_2 + i
    if st1 and st2:
        return tag_1, tag_2, True
    else:
        return False, False, False


def lemmatization(tags):
    """
    lemmatization(tags) - вход: list, выход: list

    Перевод слов к начальной форме

    Входной list:
    tags - набор тегов

    Выходной list:
    lemms - набор тегов, приведенных к начальной форме
    """
    lemms = []
    for tag in tags:
        lemms.append(morph.parse(tag)[0][2])
    return lemms


def completion(keys_annotation_tags, project_tags):
    """
    completion(keys_annotation_tags,project_tags) - вход: list, выход: list

    Дополнение набора тегов нулями до единого размера

    Входной list:
    keys_annotation_tags - теги аннотации
    project_tags - теги проекта

    Выходноой list:
    keys_annotation_tags, project_tags единой размерности
    """

    max_len = max(len(keys_annotation_tags), len(project_tags))
    if max_len - len(keys_annotation_tags) > 0:
        keys_annotation_tags += ['0' for i in range(max_len - len(keys_annotation_tags))]
    if max_len - len(project_tags) > 0:
        project_tags += ['0' for i in range(max_len - len(project_tags))]
    return keys_annotation_tags, project_tags


def change_candidate_score(keys_annotation_tag, project_tag, candidate_tag):
    """
    change_candidate_score(keys_annotation_tag,project_tag,candidate_tag) -
    вход: list, выход: int

    Подсчитывает очки пользователя для рейтинга по семантической близости между тегами,
    а также считает общее количество совпадающих тегов

    Входной list:
    keys_annotation_tag - тег аннотации
    project_tag - тег проекта
    candidate_tag - тег кандидата

    Выходной int:
    score - значение, на котрое меняется оценка пользователя в рейтинге (candidate_score)
    tags_correct - значение, на которое меняется количество совпадающих тегов (tags_correct)
    """
    proximity_1 = get_proximity(keys_annotation_tag, candidate_tag['name'])
    score = 0
    tags_correct = 0
    if proximity_1 == 'Not Found':
        pass
    elif proximity_1 > semantic_const:
        score += proximity_1 * (candidate_tag['stage']*10)**2 * additional_weight_keys_annotation
        tags_correct += 1
    if project_tag == False:
        return score, tags_correct

    proximity_2 = get_proximity(project_tag, candidate_tag['name'])
    if proximity_2 == 'Not Found':
        pass
    elif proximity_2 > semantic_const:
        score += proximity_2 * (candidate_tag['stage']) * additional_weight_project_tag
        tags_correct += 1
    return score, tags_correct


def sentiments(text):
    """
    sentiments(text) - вход: str, выход: float
  
    Анализ эмоциональной тональности текста на основе модели FastText
  
    Входной str:
    text - анализируемый текст
  
    Выходной float:
    positive - процент позитивной тональности
    negative - процент негативной тональности
    """
    res = model_sent.predict(text, k=2)[0]
    positive = 0
    negative = 0
    if 'positive' in res:
        positive = res['positive']
    if 'negative' in res:
        negative = res['negative']

    return positive, negative

def antiplag(text):
    """
    antiplag(text) - вход: str, выход: float
    
    Анализ текста на уникальность в процентах
    Используется в анализе статей. 
    Подробнее про API на сайте text.ru
    
    Входной str:
    text - анализируемый текст
  
    Выходной float:
    unique_p - процент оригинальности
    """
    your_token = ''
    z = requests.post('http://api.text.ru/post',data={'text':text[:100000],'userkey':your_token})
    unique_p =  = requests.post('http://api.text.ru/post',data={'uid':z.json()','userkey':your_token}).json()['unique']
    return unique_p
                                                                
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='883')
