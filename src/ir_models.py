import ir_datasets
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytrec_eval
import sys
import pandas as pd
import re
import nltk
import math

from whoosh import fields, index, qparser
from whoosh.analysis import StandardAnalyzer, SimpleAnalyzer, StemmingAnalyzer, RegexTokenizer, LanguageAnalyzer
from whoosh.qparser import QueryParser, OrGroup, MultifieldParser, AndGroup
from whoosh.scoring import TF_IDF, BM25F, PL2, Frequency
from collections import Counter, defaultdict, namedtuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, ndcg_score


nltk.download('stopwords')
nltk.download('punkt')


def indexing(D, preprocessing="stemming"):
    """
    Indexes documents by merging 'title', 'abstract', and 'body' into a single field,
    allowing different preprocessing strategies.

    @param D: Dataset object (from `ir_datasets`).
    @param preprocessing: The type of text preprocessing strategy to apply.
                          Options: "raw", "standard", "stemming".
    @return: Whoosh index object.
    """
    start_time = time.time()

    indexmf_dir = f"indexmf_{preprocessing}_dir"

    if os.path.exists(indexmf_dir):
        shutil.rmtree(indexmf_dir)
    os.mkdir(indexmf_dir)

    if preprocessing == "raw":
      schema = fields.Schema(
        id=fields.TEXT(stored=True),
        title=fields.TEXT(stored=True, analyzer=RegexTokenizer()),
        abstract=fields.TEXT(stored=True, analyzer=RegexTokenizer()),
        date=fields.TEXT(stored=True))
    elif preprocessing == "standard":
      schema = fields.Schema(
        id=fields.TEXT(stored=True),
        title=fields.TEXT(stored=True),
        abstract=fields.TEXT(stored=True),
        date=fields.TEXT(stored=True))
    elif preprocessing == "stemming":
      schema = fields.Schema(
        id=fields.TEXT(stored=True),
        title=fields.TEXT(stored=True, analyzer=StemmingAnalyzer()),
        abstract=fields.TEXT(stored=True, analyzer=StemmingAnalyzer()),
        date=fields.TEXT(stored=True))
    elif preprocessing == "language":
      schema = fields.Schema(
        id=fields.TEXT(stored=True),
        title=fields.TEXT(stored=True, analyzer=LanguageAnalyzer("en")),
        abstract=fields.TEXT(stored=True, analyzer=LanguageAnalyzer("en")),
        date=fields.TEXT(stored=True)) 
    else:
        raise ValueError("Invalid preprocessing option. Choose 'raw', 'standard', 'stemming' or 'language'.")


    ixmf = index.create_in(indexmf_dir, schema)
    writer = ixmf.writer()

    i=0
    for doc in D.docs_iter():
        title = str(doc[1]) if doc[1] else ""   # Title
        abstract = str(doc[4]) if doc[4] else ""  # Abstract
        date = str(doc[3]) if doc[3] else "" #Date
        

        writer.add_document(id=str(doc[0]), title=title, abstract=abstract, date=date)

    writer.commit()

    indexing_time = time.time() - start_time

    index_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(indexmf_dir)
        for f in filenames
    ) / (1024 * 1024)

    print("\nIndexing Completed!")
    print(f"Indexing Time: {indexing_time:.2f} seconds")
    print(f"Index Size: {index_size:.2f} MB")

    return ixmf, indexing_time, index_size


def ranking(I, query, model="bm25", top_k=1000):
    retrieved_results = {}
    results = {}
    
    if model.lower() == "tf":
      weighting_model = Frequency()
      boosts = None
    elif model.lower() == "tf-idf":
      weighting_model = TF_IDF()
      boosts = None
    elif model.lower() == "bm25":
      weighting_model = BM25F()
      boosts = None
    elif model.lower() == "bm25f":
      weighting_model = BM25F()
      boosts = {"title": 1, "abstract": 3}
    elif model.lower() == "lm":
        weighting_model = PL2()
        boosts = None
    else:
        raise ValueError("Invalid model. Choose 'tf-idf', 'bm25', 'bm25f', or 'lm'.")

    with I.searcher(weighting=weighting_model) as searcher:
        qp = MultifieldParser(["title","abstract"], I.schema, fieldboosts=boosts, group=OrGroup)
        parsed_query = qp.parse(query[1])
        retrieved_results = searcher.search(parsed_query, limit=top_k)
        for hit in retrieved_results:
            doc_id = hit["id"]
            doc = searcher.stored_fields(searcher.document_number(id=doc_id))
            date_str = doc.get("date", "")
            year = int(str(date_str)[:4]) if date_str and str(date_str)[:4].isdigit() else 1980
            decay = 0.1 * math.exp(-0.1 * (2021 - year))
            results[doc_id] = hit.score * decay

    return results


