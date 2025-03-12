

import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/Attack_on_Titan_(TV_series)'

response = requests.get(url)

html_content = response.text

soup = BeautifulSoup(html_content,'html.parser')

print(html_content[:500])

links = soup.find_all('a')

for  link in links:
    print(link.text)


