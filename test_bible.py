import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import bible


class TestBibleIndexing(unittest.TestCase):

    def test_bible_codes_mapping(self):
        self.assertEqual(bible.bible_codes.get('GEN'), '創世紀')
        self.assertEqual(bible.bible_codes.get('REV'), '啟示錄')
        self.assertEqual(bible.bible_codes.get('MAT'), '馬太福音')
        self.assertEqual(bible.bible_codes.get('PSA'), '詩篇')

    @patch('bible.SentenceTransformer')
    def test_sentence_transformer_embeddings(self, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        mock_model.encode.side_effect = lambda texts, **kwargs: (
            [[0.1, 0.2]] * len(texts) if isinstance(texts, list) else [0.1, 0.2]
        )

        embeddings = bible.SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        doc_vectors = embeddings.embed_documents(['text1', 'text2'])
        query_vector = embeddings.embed_query('query')

        self.assertEqual(len(doc_vectors), 2)
        self.assertEqual(doc_vectors[0], [0.1, 0.2])
        self.assertEqual(query_vector, [0.1, 0.2])

    @patch('bible.load_bible_dataset')
    def test_get_bible_documents(self, mock_load_dataset):
        fake_dataset = {
            'train': [
                {
                    'ref': ['GEN 1:1'],
                    'translation': {
                        'language': ['cmn', 'eng'],
                        'translation': ['起初，上帝創造天地。', 'In the beginning God created the heaven and the earth.']
                    },
                    'files': {
                        'file': ['cmn-cmn-cu89t.txt', 'eng-eng-kjv.txt']
                    }
                }
            ]
        }
        mock_load_dataset.return_value = fake_dataset

        docs = bible.get_bible_documents()
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIsInstance(doc, Document)
        self.assertIn('[創世紀 1:1 (GEN 1:1)]', doc.page_content)
        self.assertIn('中文: 起初，上帝創造天地。', doc.page_content)
        self.assertIn('English: In the beginning God created the heaven and the earth.', doc.page_content)
        self.assertEqual(doc.metadata['ref'], 'GEN 1:1')
        self.assertEqual(doc.metadata['book'], '創世紀')
        self.assertEqual(doc.metadata['book_code'], 'GEN')
        self.assertEqual(doc.metadata['verse'], '1:1')

    @patch('bible.load_dataset')
    def test_load_bible_dataset_fallback(self, mock_load_dataset):
        mock_load_dataset.side_effect = [Exception("Language not found"), {"train": []}]
        ds = bible.load_bible_dataset()
        self.assertEqual(ds, {"train": []})
        self.assertEqual(mock_load_dataset.call_count, 2)

    @patch('bible.QdrantVectorStore')
    @patch('bible.get_bible_documents')
    @patch('bible.SentenceTransformerEmbeddings')
    def test_main(self, mock_embeddings_cls, mock_get_docs, mock_qdrant_cls):
        mock_get_docs.return_value = [Document(page_content='sample')]
        mock_qdrant_cls.from_documents.return_value = 'mock_db'

        bible.main()

        mock_embeddings_cls.assert_called_once_with('all-MiniLM-L6-v2')
        mock_get_docs.assert_called_once()
        mock_qdrant_cls.from_documents.assert_called_once()


if __name__ == '__main__':
    unittest.main()
