import requests
import json


def get_user_profile(user_id, channel_access_token):
  url = f'https://api.line.me/v2/bot/profile/{user_id}'
  headers = {
    'Authorization': f'Bearer {channel_access_token}'
  }
  r = requests.get(url, headers=headers)
  return json.loads(r.content)
