from django.test import SimpleTestCase
from django.urls import reverse,resolve
from HAI.views import index,home,delete,predictUploaded,predict,videos,help

class TestUrls(SimpleTestCase):

    def test_index_url_resolves(self):
        #assert 1==2
        url=reverse('index')
        self.assertEquals(resolve(url).func,index)
        print('Test : test_index_url_resolves')

    def test_home_url_resolves(self):
        url=reverse('home')
        self.assertEquals(resolve(url).func,home)
        print('Test : test_home_url_resolves')

    def test_predict_url_resolves(self):
        url=reverse('predict')
        self.assertEquals(resolve(url).func,predict)
        print('Test : test_predict_url_resolves')

    def test_videos_url_resolves(self):
        url=reverse('videos')
        self.assertEquals(resolve(url).func,videos)
        print('Test : test_videos_url_resolves')

    def test_predictUploaded_url_resolves(self):
        url=reverse('predictUploaded')
        self.assertEquals(resolve(url).func,predictUploaded)
        print('Test : test_predictUploaded_url_resolves')


    def test_help_url_resolves(self):
        url=reverse('help')
        self.assertEquals(resolve(url).func,help)
        print('Test : test_help_url_resolves')

    def test_delete_url_resolves(self):
        url=reverse('delete')
        self.assertEquals(resolve(url).func,delete)
        print('Test : test_delete_url_resolves')
    