from flask import Flask
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
from rutermextract import TermExtractor
term_extractor = TermExtractor()

nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
pronouns = ' '.join(open('stopwordsadd.txt','r').read().split('\n')).split(' ,')
russian_stopwords += pronouns

morph = pymorphy2.MorphAnalyzer()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


model_url = 'http://vectors.nlpl.eu/repository/11/180.zip' # Множество вариантов моделей для использования, можно тестировать
model_file = model_url.split('/')[-1]


if not(os.path.isfile(model_file)): 
  m = wget.download(model_url)

with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

tokenizer = RegexTokenizer()

model_url = 'https://storage.b-labs.pro/models/fasttext-social-network-model.bin'
model_file = model_url.split('/')[-1]

if not(os.path.isfile(model_file)): 
  m = wget.download(model_url)

tokenizer = RegexTokenizer()
FastTextSocialNetworkModel.MODEL_PATH = 'fasttext-social-network-model.bin'
model_sent = FastTextSocialNetworkModel(tokenizer=tokenizer) 

tags_correct_const = 4
semantic_const = 0.4
ratio_const = 0.1
additional_weight_keys_annotation = 1
additional_weight_project_tag = 1
count_projects_weight = 5
count_articles_weight = 10
candidate_rating_weight = 1
citation_index_weight = 1

@app.route("/get_candidates", methods=['POST'])
@cross_origin()
def get_candidates():
    json = request.get_json()
    project_tags = json['project_tags']
    annotation = json['annotation']
    candidates = json['candidates']

    keys_annotation_tags = get_key_phrases(annotation)

    keys_annotation_tags = lemmatization(keys_annotation_tags)
    project_tags = lemmatization(project_tags)

    short_list = {}

    for candidate in candidates:
      candidate_ID = candidate['ID']
      candidate_tags = candidate['tags']
      articles = candidate['articles']
      count_articles = len(articles)
      candidate_rating = candidate['rating']
      count_projects = len(candidate['projects'])
      candidate_score = max(count_projects,0.000001)/count_projects_weight + max(candidate_rating,0.000001)/candidate_rating_weight + max(count_articles,0.000001)/count_articles_weight
      tags_correct = 0
      keys_annotation_tags,project_tags = completion(keys_annotation_tags,project_tags)
      
      for keys_annotation_tag,project_tag in zip(keys_annotation_tags,project_tags):
        for candidate_tag in candidate_tags:
          x,y = change_candidate_score(keys_annotation_tag,project_tag,candidate_tag)
          candidate_score += x; tags_correct += y

        if tags_correct > tags_correct_const:
          for article in articles:
            key_phrases_article = get_key_phrases(article)
            for key_phrase_article in key_phrases_article:
              x,y = change_candidate_score(key_phrase_article,False,candidate_tag)
              candidate_score += x; tags_correct += y
            
      short_list[candidate_ID] = candidate_score
    return sorted(short_list.items(), key=lambda x:(x[1], x[0]))[::-1]
      

      

def get_proximity(tag_1,tag_2):
    tag_1,tag_2,stage = tags_in_model(tag_1,tag_2)
    if stage:
      return model.similarity(tag_1, tag_2)
    else:
      return 'Not Found'

def get_summary(text):
  return summarize(text, ratio=ratio_const)

def get_key_phrases(text):
  text = get_summary(text)
  x = text.split('\n')
  x = [i for i in ' '.join(x).split()]
  x = term_extractor(' '.join(x))
  key_phrases = []
  for i in x:
    if morph.parse(i)[0][2] not in russian_stopwords:
      key_phrases.append(morph.parse(i)[0][2])
  return key_phrases

def tags_in_model(tag_1,tag_2):
    st1,st2 = 0,0
    for i in ['_VERB','_ADJ','_NOUN']:
      if tag_1 + i in model and not(st1):
        st1 = 1
        tag_1 = tag_1 + i
      if tag_2 + i in model and not(st2):
        st2 = 1
        tag_2 = tag_2 + i
    if st1 and st2:
      return tag_1,tag_2,True
    else:
      return False,False,False

def lemmatization(tags):
    lemms = []
    for tag in tags:
      lemms.append(morph.parse(tag)[0][2])
    return lemms

def completion(keys_annotation_tags,project_tags):
      max_len = max(len(keys_annotation_tags),len(project_tags))
      if max_len - len(keys_annotation_tags) > 0:
        keys_annotation_tags += ['0' for i in range( max_len - len(keys_annotation_tags))]
      if max_len - len(project_tags) > 0:
        project_tags += ['0' for i in range( max_len - len(project_tags))]
      return keys_annotation_tags,project_tags

def change_candidate_score(keys_annotation_tag,project_tag,candidate_tag):
      proximity_1 = get_proximity(keys_annotation_tag,candidate_tag['name'])
      score = 0
      tags_correct = 0
      if proximity_1 == 'Not Found':
        pass
      elif proximity_1>semantic_const:
        score += proximity_1 * candidate_tag['stage'] * additional_weight_keys_annotation
        tags_correct += 1
      if project_tag == False:
        return score,tags_correct

      proximity_2 = get_proximity(project_tag,candidate_tag['name'])
      if proximity_2 == 'Not Found':
        pass
      elif proximity_2>semantic_const:
        score += proximity_2 * candidate_tag['stage'] * additional_weight_project_tag
        tags_correct += 1
      return score,tags_correct
  
def sentiments(text):
  res = model_sent.predict(text, k=2)[0]
  positive = 0
  negative = 0
  if 'positive' in res:
    positive = res['positive']
  if 'negative' in res:
    negative = res['negative']

  return positive,negative


if __name__ == "__main__":
    app.run(host='0.0.0.0',port='883')
