__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# a dictionary of bible codes
bible_codes = {
  # 舊約
  'Gen': '創世紀',
  'Exo': '出埃及記',
  'Lev': '利未記',
  'Num': '民數記',
  'Deu': '申命記',
  'Jos': '約書亞記',
  'Jug': '士師記',
  'Rut': '路得記',
  '1Sa': '撒母耳記上',
  '2Sa': '撒母耳記下',
  '1Ki': '列王記上',
  '2Ki': '列王記下',
  '1Ch': '歷代志上',
  '2Ch': '歷代志下',
  'Ezr': '以斯拉記',
  'Neh': '尼希米記',
  'Est': '以斯帖記',
  'Job': '約伯記',
  'Psm': '詩篇',
  'Pro': '箴言',
  'Ecc': '傳道書',
  'Son': '雅歌',
  'Isa': '以賽亞書',
  'Jer': '耶利米書',
  'Lam': '耶利米哀',
  'Eze': '以西結書',
  'Dan': '但以理書',
  'Hos': '何西阿書',
  'Joe': '約珥書',
  'Amo': '阿摩司書',
  'Oba': '俄巴底亞',
  'Jon': '約拿書',
  'Mic': '彌迦書',
  'Nah': '那鴻書',
  'Hab': '哈巴谷書',
  'Zep': '西番雅書',
  'Hag': '哈該書',
  'Zec': '撒迦利亞',
  'Mal': '瑪拉基書',
  # 新約
  'Mat': '馬太福音',
  'Mak': '馬可福音',
  'Luk': '路加福音',
  'Jhn': '約翰福音',
  'Act': '使徒行傳',
  'Rom': '羅馬書',
  '1Co': '哥林多前',
  '2Co': '哥林多後',
  'Gal': '加拉太書',
  'Eph': '以弗所書',
  'Phl': '腓立比書',
  'Col': '歌羅西書',
  '1Ts': '帖撒羅前',
  '2Ts': '帖撒羅後',
  '1Ti': '提摩太前',
  '2Ti': '提摩太後',
  'Tit': '提多書',
  'Phm': '腓利門書',
  'Heb': '希伯來書',
  'Jas': '雅各書',
  '1Pe': '彼得前書',
  '2Pe': '彼得後書',
  '1Jn': '約翰一書',
  '2Jn': '約翰二書',
  '3Jn': '約翰三書',
  'Jud': '猶大書',
  'Rev': '啟示錄',
}


def get_bible_chapters():
  with open('bible-zh.txt', 'r') as f:
    lines = f.readlines()

    last_chapter = None
    c = []
    chapters = []
    for i, line in enumerate(lines):
      items = line.split(' ')
      code = bible_codes[items[0]]
      parts = items[1]
      chapter = parts.split(':')[0]

      if last_chapter is not None and chapter != last_chapter:
        chapter_text = '\n'.join(c)
        chapter_text = code + '\n' + chapter_text
        chapters.append(chapter_text.strip())
        c = []

      c.append(line.strip())
      last_chapter = chapter

    chapter_text = '\n'.join(c)
    chapter_text = code + '\n' + chapter_text
    chapters.append(chapter_text.strip())
  return chapters


def main():
  embeddings = OllamaEmbeddings(model='qwen:7b')
  chapters = get_bible_chapters()
  documents = [Document(page_content=c) for c in chapters]
  db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory='./bible_chroma'
  )
  print(db)


if __name__ == '__main__':
  main()
