from django.urls import path
from . import views

urlpatterns=[

    path('index',views.index,name='index'),
    path('home',views.home,name='home'),
    path('predict',views.predict,name='predict'),
    path('videos',views.videos,name='videos'),
    path('predictUploaded',views.predictUploaded,name='predictUploaded'),
    path('help',views.help,name='help'),
    path('delete',views.delete,name='delete')
    
    
]