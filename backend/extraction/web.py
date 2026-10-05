from dataclasses import dataclass
import httpx
import trafilatura
from bs4 import BeautifulSoup

@dataclass
class WebResult:
    text: str
    title: str

def process_url(url: str) -> WebResult:
    with httpx.Client(follow_redirects=True, timeout=20, headers={'User-Agent':'AccessReadAI/1.0'}) as client:
        r = client.get(url)
        r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script','style','noscript','iframe','svg','nav','footer','header','form','aside']): tag.decompose()
    for node in soup.find_all(['div','section','aside'], class_=lambda c: c and any(k in str(c).lower() for k in ['cookie','consent','advert','banner','popup','modal','newsletter','social'])): node.decompose()
    extracted = trafilatura.extract(str(soup), include_tables=True, include_links=False, favor_precision=True) or soup.get_text('\n', strip=True)
    title = soup.title.get_text(' ', strip=True) if soup.title else 'Webpage'
    return WebResult(extracted, title)
