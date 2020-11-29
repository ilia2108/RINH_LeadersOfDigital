# -*- coding: utf-8 -*-
from app import app
from app.forms import LoginForm
from flask import render_template, flash, redirect
from config import candidates
import requests
# @app.route('/')
# @app.route('/index')
# def index():
#     return  render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        annotation = form.ann.data
        tags = form.tags.data.split()
        res = requests.post('http://localhost:883/get_candidates', headers={'Content-Type': 'application/json'},
                      json={'project_tags': tags, 'candidates': candidates, 'annotation': annotation}).json()['result']
        print(res)
        list_cand = []
        for i in res:
            list_cand.append(candidates[i[0]-1])
        return render_template('final.html', title='Найти людей', form=form, res = list_cand)
    return render_template('index.html', title='Найти людей', form=form)