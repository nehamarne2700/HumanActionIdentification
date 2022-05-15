from django.test import TestCase,Client
from django.urls import reverse
import json

class TestViews(TestCase):

    def setUp(self):
        self.client=Client()
        self.index_url=reverse('index')
        self.home_url=reverse('home')
        self.predictUploaded_url=reverse('predictUploaded')
        self.predict_url=reverse('predict')
        self.help_url=reverse('help')
        self.delete_url=reverse('delete')
        self.videos_url=reverse('videos')


    def test_index_GET(self):
        responce=self.client.get(self.index_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'index.html')
        print('Test : test_index_GET')

    def test_home_GET(self):
        responce=self.client.get(self.home_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'home.html')
        print('Test : test_home_GET')

    def test_delete(self):
        responce=self.client.post(self.delete_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'video.html')
        print('Test : test_delete')

    def test_video(self):
        responce=self.client.post(self.videos_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'video.html')
        print('Test : test_video')

    def test_predict(self):
        responce=self.client.post(self.predict_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'predResult.html')
        print('Test : test_predict')

    def test_help_GET(self):
        responce=self.client.get(self.help_url)

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'help.html')
        print('Test : test_help_GET')
    
    def test_predictUploaded_POST(self):
        responce=self.client.post(self.predictUploaded_url,{
            'sel':'media\HAI\Input_Check.mp4'
        })

        self.assertEquals(responce.status_code,200)
        self.assertTemplateUsed(responce,'predResult.html')
        print('Test : test_predictUploaded_POST')

    
    