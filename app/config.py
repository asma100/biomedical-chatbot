import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'your_secret_key'
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:0@localhost/chatbot'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
