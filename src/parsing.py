# parsing.py
# Код для сбора постов из групп ВКонтакте.
# Автор: Евгений Уланов 

import time
import requests
import pandas as pd


# ----------------------------------------------------------
# Функция для групп, у которых есть domain (обычный адрес)
# ----------------------------------------------------------

def take_posts(groups_df, token):
    version = "5.199"
    count = 100
    all_posts_data = []

    for index, row in groups_df.iterrows():
        domain = row['domain_group']
        name = row['name']
        region = row['region']

        offset = 0
        while offset <= 500:
            response = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": token,
                    "v": version,
                    "domain": domain,
                    "count": count,
                    "offset": offset
                }
            )

            data = response.json().get("response", {}).get("items", [])

            for post in data:
                all_posts_data.append({
                    "name": name,
                    "region": region,
                    "text": post.get("text", ""),
                    "likes": post.get("likes", {}).get("count", 0),
                    "reposts": post.get("reposts", {}).get("count", 0),
                    "date": pd.to_datetime(post.get("date", None), unit="s")
                })

            offset += 100
            time.sleep(0.5)

    return pd.DataFrame(all_posts_data)


# ----------------------------------------------------------
# Функция для групп, у которых вместо domain есть owner_id
# ----------------------------------------------------------

def take_posts_id(groups_df, token):
    version = "5.199"
    count = 100
    all_posts_data = []

    for index, row in groups_df.iterrows():
        owner_id = row['owner_id']
        name = row['name']
        region = row['region']

        offset = 0
        while offset <= 500:
            response = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": token,
                    "v": version,
                    "owner_id": owner_id,
                    "count": count,
                    "offset": offset
                }
            )

            data = response.json().get("response", {}).get("items", [])

            for post in data:
                all_posts_data.append({
                    "name": name,
                    "region": region,
                    "text": post.get("text", ""),
                    "likes": post.get("likes", {}).get("count", 0),
                    "reposts": post.get("reposts", {}).get("count", 0),
                    "date": pd.to_datetime(post.get("date", None), unit="s")
                })

            offset += 100
            time.sleep(0.5)

    return pd.DataFrame(all_posts_data)


# ----------------------------------------------------------
# Функция, которая объединяет обе выборки
# ----------------------------------------------------------

def collect_all(groups_domain, groups_id, token):
    df1 = take_posts(groups_domain, token)
    df2 = take_posts_id(groups_id, token)
    return pd.concat([df1, df2], ignore_index=True)
