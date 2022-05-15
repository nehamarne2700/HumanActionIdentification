from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse
import time

class TestProjectBlankPage(StaticLiveServerTestCase):

    def setUp(self):
        self.browser=webdriver.Chrome('functional_tests/chromedriver.exe')

    def tearDown(self):
        self.browser.close()

    def test_dashboard_is_displayed(self):
        self.browser.get(self.live_server_url+reverse('home'))
        time.sleep(10)

    def test_dashboard_button_redirects_to_home_page(self):
        self.browser.get(self.live_server_url+reverse('home'))

        home_url=self.live_server_url + reverse('home')
        self.browser.find_element_by_class_name('dashboard').click()
        self.assertEquals(
            self.browser.current_url,
            home_url
        )

    def test_index_button_redirects_to_index_page(self):
        self.browser.get(self.live_server_url+ reverse('home'))

        index_url=self.live_server_url + reverse('index')
        button=self.browser.find_element_by_class_name('index').click()
        self.assertEquals(
            self.browser.current_url,
            index_url
        )

    def test_help_button_redirects_to_help_page(self):
        self.browser.get(self.live_server_url+ reverse('home'))

        help_url=self.live_server_url + reverse('help')
        self.browser.find_element_by_class_name('help').click()
        self.assertEquals(
            self.browser.current_url,
            help_url
        )

        

    
