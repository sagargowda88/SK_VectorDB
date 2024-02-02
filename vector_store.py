# vector_store.py
import hashlib
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class VectorStore:
    def __init__(self, embedding_model='paraphrase-MiniLM-L6-v2'):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = None  # Initialize to None, will be set during indexing
        self.embedding_model = SentenceTransformer(embedding_model)

    def create_1d_string_list(self, data, cols):
        data_rows = data[cols].astype(str).values
        return [" ".join(row) for row in data_rows]

    def get_data_hash(self, data):
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def load_embeddings_from_db(self, data_hash):
        try:
            with open(f"./embedding_storage/{data_hash}.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_embeddings_to_db(self, data_hash, embeddings):
        with open(f"./embedding_storage/{data_hash}.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    def index_data(self, data, cols):
        data_hash = self.get_data_hash(data)
        cached_embeddings = self.load_embeddings_from_db(data_hash)
        if cached_embeddings is not None:
            self.vector_data = cached_embeddings[0]
            self.vector_index = faiss.deserialize_index(cached_embeddings[1])
            return

        data_1d = self.create_1d_string_list(data, cols)
        vectors = [self.embedding_model.encode(text) for text in data_1d]
        self.vector_data = {uid: vector for uid, vector in enumerate(vectors)}
        self._build_index()

        self.save_embeddings_to_db(data_hash, (self.vector_data, faiss.serialize_index(self.vector_index)))

    def _build_index(self):
        vectors = list(self.vector_data.values())
        self.vector_index = faiss.IndexFlatIP(len(vectors[0]))
        self.vector_index.add(np.vstack(vectors))

    def find_similar_vectors(self, query_text, num_results=5):
        query_vector = self.embedding_model.encode(query_text)
        query_vector = np.expand_dims(query_vector, axis=0)

        _, indices = self.vector_index.search(query_vector, num_results)
        
        results = []
        for idx in indices[0]:
            vector_id = list(self.vector_data.keys())[idx]
            similarity = np.dot(query_vector.flatten(), self.vector_data[vector_id].flatten()) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(self.vector_data[vector_id])
            )
            results.append((vector_id, similarity))

        return results
