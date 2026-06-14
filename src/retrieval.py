import pickle
import pandas as pd


class SimpleRetrieval:
    def __init__(self, vectorizer_path, retrieval_model_path, articles_path):
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.model = pickle.load(open(retrieval_model_path, "rb"))
        self.articles = pd.read_csv(articles_path)

    def retrieve_similar_items(self, query_text, k=20):
        query_vector = self.vectorizer.transform([query_text])
        distances, indices = self.model.kneighbors(query_vector, n_neighbors=k)
        results = self.articles.iloc[indices[0]].copy()
        results["distance"] = distances[0]
        return results