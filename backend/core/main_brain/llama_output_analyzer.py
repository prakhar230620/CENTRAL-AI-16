# backend/core/output_analyzer/enhanced_llama_output_analyzer.py

import re
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
import asyncio
from backend.utils.text_to_speech.tts_service import TTSService

class EnhancedLlamaOutputAnalyzer:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sentiment_analyzer = pipeline("sentiment-analysis",
                                           model="distilbert-base-uncased-finetuned-sst-2-english")
        self.nlp = spacy.load("en_core_web_sm")
        self.topic_model = self._initialize_topic_model()
        self.fact_checker = self._initialize_fact_checker()
        self.tts_service = TTSService()

    async def analyze_output(self, user_input: str, llama_output: str) -> Tuple[bool, float, Optional[str], Optional[bytes]]:
        relevance_score = self._calculate_relevance(user_input, llama_output)
        sentiment_score = self._analyze_sentiment(llama_output)
        topic_coherence = self._check_topic_coherence(user_input, llama_output)
        factual_accuracy = self._check_factual_accuracy(llama_output)

        overall_score = (relevance_score + sentiment_score + topic_coherence + factual_accuracy) / 4

        is_relevant = overall_score >= self.confidence_threshold
        filtered_output = self._filter_output(llama_output) if is_relevant else None

        audio_data = None
        if filtered_output:
            audio_data = await self.tts_service.synthesize(filtered_output)

        return is_relevant, overall_score, filtered_output, audio_data

    def _calculate_relevance(self, user_input: str, llama_output: str) -> float:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([user_input, llama_output])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]

    def _analyze_sentiment(self, text: str) -> float:
        sentiment = self.sentiment_analyzer(text)[0]
        return sentiment['score'] if sentiment['label'] == 'POSITIVE' else 1 - sentiment['score']

    def _check_topic_coherence(self, user_input: str, llama_output: str) -> float:
        combined_text = [user_input, llama_output]
        texts = [[word for word in document.lower().split() if word not in spacy.lang.en.stop_words.STOP_WORDS]
                 for document in combined_text]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, passes=10,
                                 workers=2)

        user_topics = lda_model.get_document_topics(corpus[0])
        llama_topics = lda_model.get_document_topics(corpus[1])

        return self._calculate_topic_similarity(user_topics, llama_topics)

    def _check_factual_accuracy(self, text: str) -> float:
        inputs = self.fact_checker.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.fact_checker.model(**inputs)
        probabilities = outputs.logits.softmax(dim=-1)
        factual_score = probabilities[0][1].item()  # Assuming binary classification: [not_factual, factual]
        return factual_score

    def _filter_output(self, llama_output: str) -> str:
        doc = self.nlp(llama_output)
        filtered_sentences = []
        for sent in doc.sents:
            if not self._contains_sensitive_info(sent.text):
                filtered_sentences.append(sent.text)
        return " ".join(filtered_sentences)

    def _contains_sensitive_info(self, text: str) -> bool:
        doc = self.nlp(text)
        sensitive_entities = ["PERSON", "ORG", "GPE", "MONEY", "CREDIT_CARD", "SSN"]
        return any(ent.label_ in sensitive_entities for ent in doc.ents)

    async def route_to_other_module(self, user_input: str):
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["general_knowledge", "technical_support", "customer_service", "product_inquiry"]
        results = classifier(user_input, candidate_labels)
        top_category = results['labels'][0]

        print(f"Routing user input to {top_category} module: {user_input}")
        # Implement actual routing logic here

    def _initialize_topic_model(self):
        return LdaMulticore

    def _calculate_topic_similarity(self, topics1, topics2):
        dict1 = dict(topics1)
        dict2 = dict(topics2)

        common_topics = set(dict1.keys()) & set(dict2.keys())
        all_topics = set(dict1.keys()) | set(dict2.keys())

        if not all_topics:
            return 0.0

        return len(common_topics) / len(all_topics)

    def _initialize_fact_checker(self):
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return type('FactChecker', (), {'model': model, 'tokenizer': tokenizer})()