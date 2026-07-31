import json
import logging
import time
from pathlib import Path
import pytest
from src.agent import ask_question
from src.models import KnowledgeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestKnowledgeAssistant:

    @pytest.fixture(scope="class")
    def test_cases(self):
        with open(Path("evaluation/test_cases.json"), "r") as f:
            return json.load(f)["test_cases"]

    def get_case(self, test_cases, category):
        return next(tc for tc in test_cases if tc["category"] == category)

    def test_exact_lookup(self, test_cases):
        test = self.get_case(test_cases, "exact_lookup")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert test["expected_record_id"] in response.matched_records
        assert response.confidence >= 0.9
        assert response.needs_human_review is False

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_retrieval(self, test_cases):
        test = self.get_case(test_cases, "retrieval")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert len(response.matched_records) > 0
        assert response.confidence >= test["expected_min_confidence"]

        answer = response.answer.lower()

        assert any(
            keyword.lower() in answer
            for keyword in test["expected_keywords"]
        )

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_comparison(self, test_cases):
        test = self.get_case(test_cases, "comparison")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert len(response.matched_records) >= 2
        assert response.confidence >= test["expected_min_confidence"]

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_filtering(self, test_cases):
        test = self.get_case(test_cases, "filtering")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert len(response.matched_records) > 0
        assert response.confidence >= test["expected_min_confidence"]

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_unsupported(self, test_cases):
        test = self.get_case(test_cases, "unsupported")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert response.needs_human_review is True

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_ambiguity(self, test_cases):
        test = self.get_case(test_cases, "ambiguity")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert len(response.matched_records) > 0

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_invalid_id(self, test_cases):
        test = self.get_case(test_cases, "invalid_id")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert response.needs_human_review is True

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_mixed_query(self, test_cases):
        test = self.get_case(test_cases, "mixed_query")

        response = ask_question(test["question"])

        assert isinstance(response, KnowledgeResponse)
        assert "KB005" in response.matched_records

        logger.info(f"{test['id']} passed")
        time.sleep(1)

    def test_response_format(self):
        response = ask_question("Tell me about KB001")

        assert isinstance(response, KnowledgeResponse)
        assert isinstance(response.answer, str)
        assert isinstance(response.matched_records, list)
        assert isinstance(response.sources, list)
        assert isinstance(response.confidence, float)
        assert isinstance(response.needs_human_review, bool)

        assert len(response.answer) > 0
        assert 0.0 <= response.confidence <= 1.0

        logger.info("Response format test passed")
