"""
Basic FAISS-based RAG implementation.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding

from .base import AbstractRAG
from ..utils import load_documents_json

logger = logging.getLogger(__name__)


class BasicRAG(AbstractRAG):
    """Basic FAISS + SentenceTransformer implementation."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        domain: str = "generic"
    ):
        self.embedding_model = embedding_model
        self.model = None
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.graph = None
        self.domain = domain
        self._load_model()

    def _load_model(self) -> None:
        """Load embedding model."""
        try:
            self.model = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded SentenceTransformer model: {self.embedding_model}")
        except:
            self.model = TextEmbedding(model_name=self.embedding_model)
            logger.info(f"Loaded TextEmbedding model: {self.embedding_model}")

    def load_documents(self, json_file: str, max_tokens: int = 128) -> None:
        """Load documents from JSON file."""
        self.documents = load_documents_json(json_file, max_tokens)

    def embed_documents(self, savepath: Optional[str] = None) -> None:
        """Generate embeddings for all documents."""
        if not self.documents:
            raise ValueError("Load documents first!")

        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype(np.float32)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        logger.info(f"Created embeddings with shape {self.embeddings.shape}")
        if savepath:
            self._save_embeddings_csv(savepath)

    def load_embeddings(
        self,
        articles_csv: Optional[str] = None,
        annexes_csv: Optional[str] = None,
        recitals_csv: Optional[str] = None
    ) -> None:
        """Load pre-computed embeddings from CSV files."""
        all_docs = []

        csv_files = [
            (articles_csv, 'article'),
            (annexes_csv, 'annex'),
            (recitals_csv, 'recital')
        ]

        for csv_path, doctype in csv_files:
            if csv_path and os.path.exists(csv_path):
                all_docs.extend(self._load_csv_embeddings(csv_path, doctype))

        if not all_docs:
            raise ValueError("No embedding CSV files found or loaded!")

        self.documents = all_docs
        self.embeddings = np.array(
            [doc['embedding'] for doc in all_docs],
            dtype=np.float32
        )
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        logger.info(f"Loaded {len(self.documents)} documents from CSV embeddings")

    def _load_csv_embeddings(self, csvpath: str, doctype: str) -> List[Dict[str, Any]]:
        """Load embeddings from a single CSV file."""
        docs = []
        with open(csvpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                embedding = np.array(json.loads(row['embedding']), dtype=np.float32)
                docs.append({
                    'id': f"{doctype.capitalize()}{row['number']}",
                    'type': row['type'],
                    'number': row['number'],
                    'text': row['text'],
                    'title': f"{doctype.capitalize()} {row['number']}",
                    'embedding': embedding
                })
        logger.info(f"Loaded {len(docs)} {doctype}s from {csvpath}")
        return docs

    def _save_embeddings_csv(self, outputdir: str) -> None:
        """Save embeddings to CSV files."""
        os.makedirs(outputdir, exist_ok=True)
        docs_by_type = {'article': [], 'annex': [], 'recital': []}

        for doc in self.documents:
            if doc['type'] in docs_by_type:
                docs_by_type[doc['type']].append(doc)

        embedding_dict = {
            doc['id']: emb.tolist()
            for doc, emb in zip(self.documents, self.embeddings)
        }

        for doctype, docs in docs_by_type.items():
            if not docs:
                continue
            csvpath = os.path.join(outputdir, f"full{doctype}sembeddings.csv")
            with open(csvpath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['type', 'number', 'text', 'embedding'])
                for doc in docs:
                    embedding_str = json.dumps(embedding_dict[doc['id']])
                    writer.writerow([doctype, doc['number'], doc['text'], embedding_str])
            logger.info(f"Saved {len(docs)} {doctype}s to {csvpath}")

    def retrieve(
        self,
        query: str,
        topk: int = 20,
        threshold: float = 0.0,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if self.index is None or self.documents is None:
            raise ValueError("Index or documents missing!")

        try:
            if query_embedding is None:
                query_with_prefix = f"Represent this sentence for retrieval of relevant passages: {query}"
                query_embedding = self.model.encode(
                    [query_with_prefix],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype(np.float32)

            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            scores, indices = self.index.search(query_embedding, topk)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and float(score) >= threshold:
                    doc = self.documents[idx]
                    results.append({
                        'id': doc['id'],
                        'text': doc['text'],
                        'score': float(score),
                        'type': doc['type']
                    })

            while len(results) < topk:
                results.append({'id': 'PAD', 'text': '', 'score': -1.0, 'type': 'pad'})
            return results[:topk]

        except Exception as e:
            logger.error(f"RETRIEVE ERROR: {e}")
            return [{'id': 'ERROR', 'text': f'Error: {e}', 'score': -1.0, 'type': 'error'}] * topk

    def build_graph(self) -> None:
        """Build knowledge graph from document relationships."""
        self.graph = nx.DiGraph()
        for doc in self.documents:
            self.graph.add_node(doc['id'], type=doc['type'], text=doc['text'])

        for doc in self.documents:
            source_id = doc['id']
            if doc['type'] == 'article':
                related_recitals = doc.get('relatedRecitals', [])
                for rnum in related_recitals:
                    target_id = f"Recital_{rnum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(source_id, target_id, relation='related_recital')
            elif doc['type'] == 'annex':
                related_articles = doc.get('relatedArticles', [])
                for anum in related_articles:
                    target_id = f"Article_{anum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(source_id, target_id, relation='related_article')
                related_recitals = doc.get('relatedRecitals', [])
                for rnum in related_recitals:
                    target_id = f"Recital_{rnum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(source_id, target_id, relation='related_recital')

        logger.info(f"Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def retrieve_with_graph(
        self,
        query: str,
        topk: int = 5,
        threshold: float = 0.6,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents with graph-based expansion."""
        retrieved_docs = self.retrieve(
            query=query,
            topk=topk,
            threshold=threshold,
            query_embedding=query_embedding
        )
        retrieved_docs = [d for d in retrieved_docs if d.get("id") != "PAD"]

        final_results = []
        seen_ids = set()

        def add_doc(doc_obj: Dict[str, Any], is_graph_result: bool = False):
            if doc_obj['id'] not in seen_ids:
                seen_ids.add(doc_obj['id'])
                if is_graph_result:
                    doc_obj['graph_related'] = True
                final_results.append(doc_obj)

        for doc in retrieved_docs:
            add_doc(doc)
            if doc['type'] in ['annex', 'article'] and self.graph and self.graph.has_node(doc['id']):
                neighbors = list(self.graph.neighbors(doc['id']))
                for neighbor_id in neighbors:
                    neighbor_doc = next(
                        (d for d in self.documents if d['id'] == neighbor_id),
                        None
                    )
                    if neighbor_doc:
                        neighbor_result = {
                            'id': neighbor_doc['id'],
                            'text': neighbor_doc['text'],
                            'score': doc['score'],
                            'type': neighbor_doc['type']
                        }
                        add_doc(neighbor_result, is_graph_result=True)

        result = final_results[:topk] if final_results else [
            {'id': 'PAD', 'text': '', 'score': -1.0, 'type': 'pad'}
        ] * topk
        return result
