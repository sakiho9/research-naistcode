import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import pandas as pd

# 53ページ確認済み
max_pages = 53  # 取得するページ数
max_blog_pages = 20000  # 各ブログから取得するページ数
exclude_keywords = ["摂食障害", "拒食症", "過食症"]

from tqdm import tqdm

def scrape_ranking_items(url, max_pages):
    base_url = 'https://blogger.ameba.jp'
    items_data = []

    for page in range(max_pages):
        print(f"ページ {page + 1} を取得中...")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # ランキングリストの取得
        ranking_list = soup.find('ol', class_='p-rankingAllImage')
        if not ranking_list:
            print("ランキングリストが見つかりませんでした。")
            break

        # ランキングアイテムの抽出
        ranking_items = ranking_list.find_all('li', class_='p-rankingAllText__item')

        # tqdmで進捗バーを表示しながらループ
        for item in tqdm(ranking_items, desc="ブログ取得中"):
            title_tag = item.find('h3', class_='p-rankingAllText__title').find('a')
            title = title_tag.get_text(strip=True) if title_tag else 'N/A'
            link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else 'N/A'

            # ヘッダーとプロフィール情報の取得
            if not contains_excluded_keywords(link):
                # 各ブログの情報を取得（ブログ本文と日付とプロフィール情報を含む）
                blog_series_info = scrape_blog_series(title, link, max_blog_pages)
                items_data.extend(blog_series_info)

        # 次のページのリンクを取得
        next_link_tag = soup.find('li', class_='c-pager__item--next').find('a')
        if not next_link_tag:
            print("次のページが見つかりませんでした。")
            break
        next_link = base_url + next_link_tag['href']

        url = next_link

        # ウェイトを入れる（サーバーに負荷をかけないため）
        time.sleep(2)

    return items_data

def contains_excluded_keywords(blog_url):
    # ヘッダー情報をチェック
    if contains_excluded_keywords_in_header(blog_url):
        return True

    # プロフィール情報をチェック
    profile_url = f"https://www.ameba.jp/profile/{get_user_name(blog_url)}/"
    if contains_excluded_keywords_in_profile(profile_url):
        return True

    return False

def contains_excluded_keywords_in_header(blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # ヘッダー情報を取得
        header = soup.find('div', {'class': ['skin-headerTitle', 'skinBlogHeadingGroupArea']})
        header_text = header.get_text(strip=True) if header else ''

        # キーワードが含まれているかチェック
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

        # プロフィール情報を取得
        profile = soup.find('div', {'class': 'user-info'})
        profile_text = profile.get_text(strip=True) if profile else ''

        # キーワードが含まれているかチェック
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
            break  # 前のページと同じサブタイトルと本文が出てきたら終了
        
        if current_subtitle == 'N/A' or current_content == 'N/A':
            break  # 終了条件：ブログ情報が見つからない場合

        blog_series_info.append(blog_info)
        last_subtitle = current_subtitle
        last_content = current_content
        page_num += 1
        time.sleep(2)  # サーバーに負荷をかけないためのウェイト

    return blog_series_info

def scrape_blog_info(title, blog_url):
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # ブログのタイトルを取得
        if soup.find('div', class_='skinArticleHeader2'):  # div の場合
            blog_title_tag = soup.find('div', class_='skinArticleHeader2').find('h1').find('a')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        elif soup.find('h2', class_='skin-entryTitle'):  # h2 の場合
            blog_title_tag = soup.find('h2', class_='skin-entryTitle').find('a', class_='skinArticleTitle')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        elif soup.find('h1', class_='skinArticleTitle'):  # h1 の場合
            blog_title_tag = soup.find('h1', class_='skinArticleTitle').find('a', class_='skinArticleTitle')
            blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else 'N/A'
        else:
            blog_title = 'N/A'

        # ブログの投稿日を取得
        post_date = scrape_blog_post_date(blog_url)

        # ブログ本文を取得
        blog_content = scrape_blog_content(blog_url)

        # プロフィール情報を取得
        profile_info = scrape_profile_info(blog_url)
        # ブログの情報を辞書として返す
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

        # ブログの投稿日時を取得
        if soup.find('time', class_='skin-textQuiet'):
            post_date_element = soup.find('time', class_='skin-textQuiet')
            if post_date_element.has_attr('datetime'):
                return post_date_element['datetime'].split('T')[0]

        # 'entry_created_datetime'が含まれるテキストを検索して日付を抽出
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

        # ブログ本文を含む要素を特定する（例として、指定されたHTMLを参照）
        blog_content = soup.find('div', {'data-google-interstitial': 'false', 'class': ['articleText', '_2nkwn0s9']})

        if blog_content:
            # ブログ本文を抜き出す
            blog_text = blog_content.get_text(separator=' ').strip()  # 改行せずに一行で表示
            return blog_text

        else:
            return 'N/A'

    except requests.exceptions.RequestException as e:
        print(f"{blog_url} からブログ本文を取得中にエラーが発生しました: {e}")
        return 'N/A'

def scrape_profile_info(blog_url):
    try:
        # プロフィールページのURLを構築
        profile_url = f"https://www.ameba.jp/profile/{get_user_name(blog_url)}/"
        
        # プロフィールページを取得
        response = requests.get(profile_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # プロフィール情報を取得
        profile_element = soup.find('div', class_='user-info__detail -clearfix')

        # 要素が存在するかを確認
        if profile_element:
            # プロフィール情報をリストに分割
            profile_info = profile_element.find_all('dl', class_='user-info__list')
            
            # 性別と生年月日の情報を初期化
            gender = "NaN"
            birth_date = None  # 生年月日を初期化

            # プロフィール情報から性別と生年月日を抽出
            for info in profile_info:
                term = info.find('dt', class_='user-info__term').text.strip()
                value = info.find('dd', class_='user-info__value').text.strip()
                if term == "性別":
                    gender = value
                elif term == "生年月日":
                    birth_date = datetime.strptime(value, "%Y年%m月%d日")

            # 年齢を計算
            if birth_date:
                today = datetime.now()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            else:
                age = "NaN"

        else:
            gender = "NaN"
            age = "NaN"

        return {'profile_gender': gender, 'profile_age': age, 'user_name': get_user_name(blog_url)}

    except requests.exceptions.RequestException as e:
        print(f"{profile_url} からプロフィール情報を取得中にエラーが発生しました: {e}")
        return {'profile_gender': 'N/A', 'profile_age': 'N/A', 'user_name': get_user_name(blog_url)}

def get_user_name(blog_url):
    # ブログのURLからユーザ名を抽出する関数
    return blog_url.split('/')[3]

# ランキングページのURL
url = 'https://blogger.ameba.jp/genres/diet/blogs/ranking'
items = scrape_ranking_items(url, max_pages)

# 各ブログの情報を取得して表示
if items:
    for item in items:
        print(f"ブログタイトル: {item['blog_title']}")
        print(f"リンク: {item['url']}")
        print(f"タイトル: {item['title']}")
        print(f"投稿日: {item['date']}")
        print(f"ブログ内容: {item['text'][:50]}...")  # 本文の先頭部分を表示（50文字まで）
        print(f"プロフィール性別: {item['sex']}")
        print(f"プロフィール年齢: {item['age']}")
        print(f"ユーザ名: {item['user']}")
        print('-' * 50)
else:
    print("要素が見つかりませんでした。")


# データをCSVファイルに保存
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# 収集したデータを保存するための関数
save_to_csv(items, 'diet_blogs_new4.csv')