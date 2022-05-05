from django.urls import path
from . import views

urlpatterns=[

    path('',views.index,name='index'),
    path('predict',views.predict,name='predict'),
    path('videos',views.videos,name='videos'),
    path('predictUploaded',views.predictUploaded,name='predictUploaded')
    
    
]