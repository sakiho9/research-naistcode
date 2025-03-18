import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

max_blog_pages = 20000  # 各ブログから取得するページ数
exclude_keywords = ["摂食障害", "拒食症", "過食症"]
csv_filename = 'blogs_14.csv'

def scrape_ranking_items(url):
    base_url = 'https://blogger.ameba.jp'
    items_data = []

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    ranking_list = soup.find('ol', class_='p-rankingAllImage')
    if not ranking_list:
        print("ランキングリストが見つかりませんでした。")
        return items_data

    ranking_items = soup.find_all('li', class_='p-rankingAllText__item')
    urls = []
    titles = []
    for item in ranking_items:
        title_tag = item.find('h3', class_='p-rankingAllText__title').find('a')
        title = title_tag.get_text(strip=True) if title_tag else 'N/A'
        link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'N/A'
        if not contains_excluded_keywords(link):
            urls.append(link)
            titles.append(title)

    # 収集するデータに対してtqdmの進捗バーを表示
    for title, url in tqdm(zip(titles, urls), total=len(urls), desc="ブログ取得中"):
        blog_series_info = scrape_blog_series(title, url, max_blog_pages)
        items_data.extend(blog_series_info)
        
    # 収集したデータをCSVファイルに保存
    df = pd.DataFrame(items_data)
    df.to_csv(csv_filename, index=False)

    return items_data

def contains_excluded_keywords(blog_url):
    if contains_excluded_keywords_in_header(blog_url):
        return True

    profile_url = f"https://www.ameba.jp/profile/{get_user_name(blog_url)}/"
    if contains_excluded_keywords_in_profile(profile_url):
        return True

    return False

def contains_excluded_keywords_in_header(blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        header = soup.find('div', {'class': ['skin-headerTitle', 'skinBlogHeadingGroupArea']})
        header_text = header.get_text(strip=True) if header else ''

        for keyword in exclude_keywords:
            if keyword in header_text:
                return True
        return False

    except requests.exceptions.RequestException as e:
        print(f"{blog_url} からヘッダー情報を取得中にエラーが発生しました: {e}")
        return False

def contains_excluded_keywords_in_profile(profile_url):
    try:
        response = requests.get(profile_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        profile = soup.find('div', {'class': 'user-info'})
        profile_text = profile.get_text(strip=True) if profile else ''

        for keyword in exclude_keywords:
            if keyword in profile_text:
                return True
        return False

    except requests.exceptions.RequestException as e:
        print(f"{profile_url} からプロフィール情報を取得中にエラーが発生しました: {e}")
        return False

def scrape_blog_series(title, blog_url, max_blog_pages):
    blog_series_info = []
    page_num = 1
    last_subtitle = None
    last_content = None

    while page_num <= max_blog_pages:
        url = f"{blog_url}page-{page_num}.html" if page_num > 1 else blog_url
        blog_info = scrape_blog_info(title, url)
        
        current_subtitle = blog_info.get('title', 'N/A')
        current_content = blog_info.get('text', 'N/A')
        
        if current_subtitle == last_subtitle and current_content == last_content:
            break
        
        if current_subtitle == 'N/A' or current_content == 'N/A':
            break

        blog_series_info.append(blog_info)
        last_subtitle = current_subtitle
        last_content = current_content
        page_num += 1
        time.sleep(2)

    return blog_series_info

def scrape_blog_info(title, blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        if soup.find('div', class_='skinArticleHeader2'):
            blog_title_tag = soup.find('div', class_='skinArticleHeader2').find('h1').find('a')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        elif soup.find('h2', class_='skin-entryTitle'):
            blog_title_tag = soup.find('h2', class_='skin-entryTitle').find('a', class_='skinArticleTitle')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        elif soup.find('h1', class_='skinArticleTitle'):
            blog_title_tag = soup.find('h1', class_='skinArticleTitle').find('a', class_='skinArticleTitle')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        else:
            blog_title = 'N/A'

        post_date = scrape_blog_post_date(blog_url)
        blog_content = scrape_blog_content(blog_url)
        profile_info = scrape_profile_info(blog_url)

        blog_info = {
            'blog_title': title,
            'sex': profile_info['profile_gender'],
            'age': profile_info['profile_age'], 
            'user': profile_info['user_name'], 
            'url': blog_url, 
            'date': post_date,
            'title': blog_title,
            'text': blog_content
        }

    except requests.exceptions.RequestException as e:
        print(f"{blog_url} からブログ情報を取得中にエラーが発生しました: {e}")
        blog_info = {
            'blog_title': title,
            'sex': 'N/A',
            'age': 'N/A',
            'user': 'N/A', 
            'url': blog_url,
            'date': 'N/A', 
            'title': 'N/A',
            'text': 'N/A',
        }

    return blog_info

def scrape_blog_post_date(blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        if soup.find('time', class_='skin-textQuiet'):
            post_date_element = soup.find('time', class_='skin-textQuiet')
            if post_date_element.has_attr('datetime'):
                return post_date_element['datetime'].split('T')[0]

        elements_with_text = soup.find_all(string=re.compile(r'entry_created_datetime'))
        for element_text in elements_with_text:
            match = re.search(r'entry_created_datetime":"(\d{4}-\d{2}-\d{2})', element_text)
            if match:
                return match.group(1)

        return 'N/A'

    except requests.exceptions.RequestException as e:
        print(f"{blog_url} から投稿日を取得中にエラーが発生しました: {e}")
        return 'N/A'

def scrape_blog_content(blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        blog_content = soup.find('div', {'data-google-interstitial': 'false', 'class': ['articleText', '_2nkwn0s9']})

        if blog_content:
            blog_text = blog_content.get_text(separator=' ').strip()
            return blog_text

        else:
            return 'N/A'

    except requests.exceptions.RequestException as e:
        print(f"{blog_url} からブログ本文を取得中にエラーが発生しました: {e}")
        return 'N/A'

def scrape_profile_info(blog_url):
    try:
        profile_url = f"https://www.ameba.jp/profile/{get_user_name(blog_url)}/"
        
        response = requests.get(profile_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        profile_element = soup.find('div', class_='user-info__detail -clearfix')

        if profile_element:
            profile_info = profile_element.find_all('dl', class_='user-info__list')
            gender = "NaN"
            birth_date = None

            for info in profile_info:
                term = info.find('dt', class_='user-info__term').text.strip()
                value = info.find('dd', class_='user-info__value').text.strip()
                if term == "性別":
                    gender = value
                elif term == "生年月日":
                    if value != "非公開":
                        try:
                            # 年月日の形式に合わせて正しいフォーマットを指定する
                            birth_date = datetime.strptime(value, "%Y年%m月%d日")
                        except ValueError as e:
                            print(f"生年月日の解析中にエラーが発生しました: {e}")

            current_date = datetime.now()
            age = current_date.year - birth_date.year if birth_date else "NaN"

            user_name = get_user_name(blog_url)

            return {'profile_gender': gender, 'profile_age': age, 'user_name': user_name}

        else:
            return {'profile_gender': 'NaN', 'profile_age': 'NaN', 'user_name': 'NaN'}

    except requests.exceptions.RequestException as e:
        print(f"{profile_url} からプロフィール情報を取得中にエラーが発生しました: {e}")
        return {'profile_gender': 'NaN', 'profile_age': 'NaN', 'user_name': 'NaN'}


def get_user_name(blog_url):
    match = re.search(r'https?://ameblo.jp/([^/]+)/', blog_url)
    if match:
        return match.group(1)
    return 'NaN'

# 実行
ranking_url = "https://blogger.ameba.jp/genres/diet/blogs/ranking?rank=237&amebaId=hirobesu1213"
scrape_ranking_items(ranking_url)