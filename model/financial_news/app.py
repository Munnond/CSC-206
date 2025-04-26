from flask import Flask, render_template, request
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
from PIL import Image
import io
import nltk
nltk.download('punkt')

app = Flask(__name__)

def fetch_rss(url):
    op = urlopen(url)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    return sp_page.find_all('item')

def fetch_news_search_topic(topic):
    return fetch_rss(f'https://news.google.com/rss/search?q={topic}')

def get_summary(link):
    try:
        news = Article(link)
        news.download()
        news.parse()
        news.nlp()
        return news.title, news.summary, news.top_image
    except:
        return None, "Summary not available", "/static/images/no_image.jpg"

@app.route('/', methods=['GET', 'POST'])
def index():
    news_list = []
    topic = request.form.get('topic')
    quantity = int(request.form.get('quantity', 5))

    if request.method == 'POST':
            news_list = fetch_news_search_topic(topic.replace(' ', ''))

    summaries = []
    for news in news_list[:quantity]:
        title, summary, image_url = get_summary(news.link.text)
        summaries.append({
            'title': news.title.text,
            'link': news.link.text,
            'summary': summary,
            'image': image_url,
            'source': news.source.text if news.source else '',
            'pubDate': news.pubDate.text
        })

    return render_template('index.html', summaries=summaries)

if __name__ == '__main__':
    app.run(debug=True)
