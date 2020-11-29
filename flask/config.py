import os

class Config(object):
    SECRET_KEY =  'ЦИФРОВОЙПРОРЫВ'
    
candidates = \
  [
  {'ID':1, 'articles': [], 'rating': 4, 'projects': [1], 'tags': [{'name': 'физиология', 'stage': 3}, {'name': 'кардиология', 'stage': 1}, {'name': 'онкология', 'stage': 2}],'citation':1,'h_index':1},
  {'ID':2,'articles':['Исследование раковых опухолей'],'rating':5,'projects':[],'tags':[{'name':'биология','stage':1},{'name':'онкология','stage':2}],'citation':1,'h_index':1},
  {'ID':3, 'articles': [], 'rating': 3, 'projects': [1], 'tags': [{'name': 'история', 'stage': 3}, {'name': 'россия', 'stage': 1}, {'name': 'обществознание', 'stage': 2}],'citation':1,'h_index':1},
  {'ID':4,'articles':['Авиастроение в современности'],'rating':3,'projects':[],'tags':[{'name':'программирование','stage':3},{'name':'спорт','stage':3}],'citation':1,'h_index':1},
  {'ID':5,'articles':[],'rating':0,'projects':[1],'tags':[{'name':'ядерная физика','stage':3},{'name':'физическая культура','stage':1},{'name':'история','stage':2}],'citation':1,'h_index':1},
  {'ID':6,'articles':[],'rating':0,'projects':[1,1,1],'tags':[{'name':'биология','stage':3},{'name':'нейрофизиология','stage':2},{'name':'машинное обучение','stage':1}],'citation':1,'h_index':1},
  {'ID':7,'articles':[],'rating':3,'projects':[1],'tags':[{'name':'обществознание','stage':3},{'name':'история','stage':2}],'citation':1,'h_index':1},
  {'ID':8,'articles':[],'rating':5,'projects':[1],'tags':[{'name':'физика','stage':3},{'name':'биология','stage':1},{'name':'авиастроение','stage':2}],'citation':1,'h_index':1},
  {'ID':9, 'articles': ['Исследование раковых опухолей', 'Разбор карбюратора'], 'rating': 3, 'projects': [], 'tags': [{'name': 'биология', 'stage': 1}, {'name': 'онкология', 'stage': 2}],'citation':1,'h_index':1},
  {'ID': 10, 'articles': ['Теория струн'], 'rating': 3, 'projects': [], 'tags': [{'name': 'биология', 'stage': 1}, {'name': 'онкология', 'stage': 2}],'citation':1,'h_index':1},
  ]