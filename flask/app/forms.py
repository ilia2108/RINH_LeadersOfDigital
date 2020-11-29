from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    ann = StringField('Аннотация к проекту', validators=[DataRequired()])
    tags = StringField('Теги (через пробел)', validators=[DataRequired()])
    search = SubmitField('Найти')