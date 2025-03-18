# ブログタイトル全て（53156件）, 日付取得込み
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import re
num = 53158

# データを格納するリスト
all_data = []

# 初期URL
base_url = 'https://search.ameba.jp/general/blogger/%E3%83%80%E3%82%A4%E3%82%A8%E3%83%83%E3%83%88.html'
# 初期ページURL
current_url = base_url

# ブログの取得数
blog_count = 0

# 日付を取得する関数
def extract_date_from_html(html_content):
    # 正規表現パターンを定義
    pattern = r'"dateModified":"(\d{4}-\d{2}-\d{2})'
    
    # パターンに一致する部分を検索
    match = re.search(pattern, html_content)
    
    if match:
        # 日付部分を取得
        date = match.group(1)
    else:
        date = "N/A"
        # print("日付が見つかりませんでした")
    
    return date

# テキスト全文とユーザ情報を取得する関数
def get_full_text_and_user_info(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    # テキスト全文の取得（<p>タグ）
    text_tags = soup.find_all('p')
    full_text_p = ' '.join(tag.get_text(strip=True) for tag in text_tags)
    
    # テキスト全文の取得（<div class="skin-entryBody _2nkwn0s9">タグ）
    text_div = soup.find('div', class_='skin-entryBody _2nkwn0s9')
    full_text_div = text_div.get_text(strip=True) if text_div else ''
    
    # articleTextの取得（<div class="articleText">タグ）
    article_text_div = soup.find('div', class_='articleText')
    article_text = article_text_div.get_text(strip=True) if article_text_div else ''
    
    # 結合して全文を取得
    full_text = f"{full_text_p} {full_text_div} {article_text}".strip()
    
    # HTMLコンテンツを文字列として取得
    html_content = response.content.decode('utf-8')
    
    # 日付の取得
    date = extract_date_from_html(html_content)

    # ユーザ情報の取得
    user_info = {}

    # ユーザ名の取得
    # 正規表現パターン
    pattern = r"https://ameblo\.jp/([^/]+)/?$"

    # マッチング
    match = re.search(pattern, link)

    if match:
        result = match.group(1)
    else:
        result = "N/A"
        print("マッチが見つかりませんでした")

    # URL を生成
    profile_url = f"https://profile.ameba.jp/ameba/{result}/"
    response = requests.get(profile_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # プロフィール情報を取得
    profile_element = soup.find('div', class_='user-info__detail -clearfix')

    # 要素が存在するかを確認
    if profile_element:
        # プロフィール情報をリストに分割
        profile_info = profile_element.find_all('dl', class_='user-info__list')
        
        # 性別と生年月日の情報を初期化
        gender = "NaN"
        birth_year = "NaN"

        # プロフィール情報から性別と生年月日を抽出
        gender = None
        birth_date = None

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

        # # 結果を出力
        # print("性別:", gender)
        # print("年代:", age)
    else:
        # print("プロフィール情報が見つかりませんでした。")
        gender = "NaN"
        age = "NaN"

    # ユーザ情報を追加
    user_info = {
        'name': result,
        'profile_gender': gender if gender is not None else "NaN",  # 追加
        'profile_age': age if age is not None else "NaN"  # 追加
    }
    
    return full_text, date, user_info

# データを取得する関数
def get_page_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = soup.find_all('li', class_='PcBloggerListItem')

    page_data = []
    for article in articles:
        title_tag = article.find('a', class_='PcBloggerListItem_BlogLink')
        title = title_tag.get_text(strip=True) if title_tag else ''
        
        link_tag = article.find('a', class_='PcBloggerListItem_BlogLink')
        link = link_tag['href'] if link_tag else ''

        # date_tag = article.find('div', class_='PcBloggerListItem_LatestEntryDate')
        # date = date_tag.get_text(strip=True) if date_tag else ''

        blog_title_tag = article.find('div', class_='PcBloggerListItem_LatestEntryDate')
        blog_title = blog_title_tag.get_text(strip=True) if blog_title_tag else ''
        
        # thumbnail_tag = article.find('img', class_='LazyImage_Loading')
        # thumbnail = thumbnail_tag['data-src'] if thumbnail_tag else ''
        
        # テキスト全文とユーザ情報を取得
        full_text, date, user_info = get_full_text_and_user_info(link)
        
        page_data.append({
            'sex': user_info.get('profile_gender'),  # 性別を追加
            'age': user_info.get('profile_age'),  # 年齢を追加
            'user': user_info.get('name'),  # ユーザ名を追加
            'url': link,
            'date': date,
            'title': title,
            # 'blog_title': blog_title,
            # 'thumbnail': thumbnail,  
            'text': full_text,  # テキスト全文を追加
        })
    
    return page_data, soup

# 「次の10件」リンクを取得する関数
def get_next_page_url(soup):
    next_link_tag = soup.find('a', class_='PcResultPagination_MoreLink')
    if next_link_tag and 'href' in next_link_tag.attrs:
        return next_link_tag['href']
    else:
        return None

# データ収集ループ
with tqdm(total=num) as pbar:
    while current_url and blog_count < num:
        pbar.set_description(f"Fetching data from {current_url}")
        page_data, soup = get_page_data(current_url)
        all_data.extend(page_data)
        next_page_url = get_next_page_url(soup)
        
        # ページ番号の取得
        current_page_number = int(parse_qs(urlparse(current_url).query).get('p', [1])[0])
        
        # 次のページのURLを生成
        next_page_number = current_page_number + 1
        next_url = base_url + f'?p={next_page_number}'
        
        current_url = next_url
        blog_count += len(page_data)

        # 次のページナンバーが5317の場合、次のページが存在しないため、ループを終了
        if next_page_number == 5317:
            break        

        # ブログの取得数がnum件を超えた場合、必要なエントリー数を削除する
        if blog_count > num:
            excess_entries = blog_count - num
            all_data = all_data[:-excess_entries]

        pbar.update(len(page_data))

# 結果を表示
for entry in all_data:
    print(f"Title: {entry['title']}")
    print(f"Link: {entry['url']}")
    print(f"Date: {entry['date']}")
    # print(f"Blog Title: {entry['blog_title']}")
    # print(f"Thumbnail: {entry['thumbnail']}")
    print(f"Full Text: {entry['text'][:500]}...")  # 最初の500文字のみ表示
    print(f"ameba id: {entry['user']}")  # ユーザ名を表示
    print(f"gender: {entry['sex']}")  # 性別を表示
    print(f"age: {entry['age']}")  # 年齢を表示
    print('-' * 50)

# データをCSVファイルに保存
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# 収集したデータを保存するための関数
save_to_csv(all_data, 'diet_blogs_new3.csv')