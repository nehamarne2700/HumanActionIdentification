from django.apps import AppConfig
from django.conf import settings
import pickle
import os

class HaiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'HAI'
    path=os.path.join(settings.MODELS,'body_language.pkl')

    with open(path,'rb') as pickled:
        data=pickle.load(pickled)

    model=data