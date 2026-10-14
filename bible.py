from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer

# a dictionary of bible codes mapping standard abbreviations and USFM codes to Chinese book names
bible_codes = {
    # 舊約
    'GEN': '創世紀', 'Gen': '創世紀',
    'EXO': '出埃及記', 'Exo': '出埃及記',
    'LEV': '利未記', 'Lev': '利未記',
    'NUM': '民數記', 'Num': '民數記',
    'DEU': '申命記', 'Deu': '申命記',
    'JOS': '約書亞記', 'Jos': '約書亞記',
    'JDG': '士師記', 'JUG': '士師記', 'Jug': '士師記',
    'RUT': '路得記', 'Rut': '路得記',
    '1SA': '撒母耳記上', '1Sa': '撒母耳記上',
    '2SA': '撒母耳記下', '2Sa': '撒母耳記下',
    '1KI': '列王記上', '1Ki': '列王記上',
    '2KI': '列王記下', '2Ki': '列王記下',
    '1CH': '歷代志上', '1Ch': '歷代志上',
    '2CH': '歷代志下', '2Ch': '歷代志下',
    'EZR': '以斯拉記', 'Ezr': '以斯拉記',
    'NEH': '尼希米記', 'Neh': '尼希米記',
    'EST': '以斯帖記', 'Est': '以斯帖記',
    'JOB': '約伯記', 'Job': '約伯記',
    'PSA': '詩篇', 'PSM': '詩篇', 'Psm': '詩篇',
    'PRO': '箴言', 'Pro': '箴言',
    'ECC': '傳道書', 'Ecc': '傳道書',
    'SNG': '雅歌', 'SON': '雅歌', 'Son': '雅歌',
    'ISA': '以賽亞書', 'Isa': '以賽亞書',
    'JER': '耶利米書', 'Jer': '耶利米書',
    'LAM': '耶利米哀', 'Lam': '耶利米哀',
    'EZK': '以西結書', 'EZE': '以西結書', 'Eze': '以西結書',
    'DAN': '但以理書', 'Dan': '但以理書',
    'HOS': '何西阿書', 'Hos': '何西阿書',
    'JOL': '約珥書', 'JOE': '約珥書', 'Joe': '約珥書',
    'AMO': '阿摩司書', 'Amo': '阿摩司書',
    'OBA': '俄巴底亞', 'Oba': '俄巴底亞',
    'JON': '約拿書', 'Jon': '約拿書',
    'MIC': '彌迦書', 'Mic': '彌迦書',
    'NAM': '那鴻書', 'NAH': '那鴻書', 'Nah': '那鴻書',
    'HAB': '哈巴谷書', 'Hab': '哈巴谷書',
    'ZEP': '西番雅書', 'Zep': '西番雅書',
    'HAG': '哈該書', 'Hag': '哈該書',
    'ZEC': '撒迦利亞', 'Zec': '撒迦利亞',
    'MAL': '瑪拉基書', 'Mal': '瑪拉基書',
    # 新約
    'MAT': '馬太福音', 'Mat': '馬太福音',
    'MRK': '馬可福音', 'MAK': '馬可福音', 'Mak': '馬可福音',
    'LUK': '路加福音', 'Luk': '路加福音',
    'JHN': '約翰福音', 'Jhn': '約翰福音',
    'ACT': '使徒行傳', 'Act': '使徒行傳',
    'ROM': '羅馬書', 'Rom': '羅馬書',
    '1CO': '哥林多前', '1Co': '哥林多前',
    '2CO': '哥林多後', '2Co': '哥林多後',
    'GAL': '加拉太書', 'Gal': '加拉太書',
    'EPH': '以弗所書', 'Eph': '以弗所書',
    'PHP': '腓立比書', 'PHL': '腓立比書', 'Phl': '腓立比書',
    'COL': '歌羅西書', 'Col': '歌羅西書',
    '1TH': '帖撒羅前', '1TS': '帖撒羅前', '1Ts': '帖撒羅前',
    '2TH': '帖撒羅後', '2TS': '帖撒羅後', '2Ts': '帖撒羅後',
    '1TI': '提摩太前', '1Ti': '提摩太前',
    '2TI': '提摩太後', '2Ti': '提摩太後',
    'TIT': '提多書', 'Tit': '提多書',
    'PHM': '腓利門書', 'Phm': '腓利門書',
    'HEB': '希伯來書', 'Heb': '希伯來書',
    'JAS': '雅各書', 'Jas': '雅各書',
    '1PE': '彼得前書', '1Pe': '彼得前書',
    '2PE': '彼得後書', '2Pe': '彼得後書',
    '1JN': '約翰一書', '1Jn': '約翰一書',
    '2JN': '約翰二書', '2Jn': '約翰二書',
    '3JN': '約翰三書', '3Jn': '約翰三書',
    'JUD': '猶大書', 'Jud': '猶大書',
    'REV': '啟示錄', 'Rev': '啟示錄',
}


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False)
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return [list(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)


def load_bible_dataset():
    try:
        return load_dataset(
            "bible-nlp/biblenlp-corpus",
            languages=["zho", "eng"],
            pair="single"
        )
    except Exception:
        return load_dataset(
            "bible-nlp/biblenlp-corpus",
            languages=["cmn", "eng"],
            pair="single"
        )


def get_bible_documents():
    dataset = load_bible_dataset()
    documents = []

    for split in dataset.keys():
        for sample in dataset[split]:
            ref_raw = sample['ref'][0] if sample.get('ref') else ''
            if not ref_raw:
                continue

            parts = ref_raw.split(maxsplit=1)
            book_code = parts[0].upper()
            verse_num = parts[1] if len(parts) > 1 else ''

            chinese_book = bible_codes.get(book_code, bible_codes.get(parts[0], book_code))

            zh_texts = []
            en_texts = []
            seen = set()

            langs = sample.get('translation', {}).get('language', [])
            texts = sample.get('translation', {}).get('translation', [])

            for lang, text in zip(langs, texts):
                text = text.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                if lang in ('cmn', 'zho', 'zh'):
                    zh_texts.append(text)
                elif lang in ('eng', 'en'):
                    en_texts.append(text)

            header = f'[{chinese_book} {verse_num} ({ref_raw})]'
            content_parts = [header]
            if zh_texts:
                content_parts.append('中文: ' + ' / '.join(zh_texts[:2]))
            if en_texts:
                content_parts.append('English: ' + ' / '.join(en_texts[:1]))

            page_content = '\n'.join(content_parts)
            doc = Document(
                page_content=page_content,
                metadata={
                    'ref': ref_raw,
                    'book': chinese_book,
                    'book_code': book_code,
                    'verse': verse_num,
                }
            )
            documents.append(doc)

    return documents


def main():
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    documents = get_bible_documents()
    db = QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        path="data/my_qdrant_db",
        collection_name="bible"
    )
    print(db)


if __name__ == '__main__':
    main()
