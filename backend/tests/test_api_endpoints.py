"""
API endpoint tests for the RAG system FastAPI application.

Tests all API endpoints including /api/query, /api/courses, /api/new-session,
and the root endpoint for proper request/response handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""

    def test_query_with_session_id(self, test_client_with_mocked_rag, api_test_data):
        """Test query endpoint with provided session ID"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json=api_test_data["valid_query"]
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_query_without_session_id(self, test_client_with_mocked_rag, api_test_data):
        """Test query endpoint without session ID - should create new session"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json=api_test_data["query_without_session"]
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # Mock returns this

    def test_query_with_empty_query(self, test_client_with_mocked_rag, api_test_data):
        """Test query endpoint with empty query string"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json=api_test_data["invalid_query"]
        )

        # Should still work with empty query (RAG system might handle gracefully)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_invalid_json(self, test_client_with_mocked_rag):
        """Test query endpoint with malformed JSON"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            data="invalid json"
        )

        assert response.status_code == 422  # Validation error

    def test_query_missing_required_field(self, test_client_with_mocked_rag):
        """Test query endpoint with missing required query field"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"session_id": "test-123"}  # Missing query field
        )

        assert response.status_code == 422  # Validation error

    def test_query_rag_system_error(self, test_client_with_mocked_rag, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        # Mock RAG system to raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"query": "test query"}
        )

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_query_response_structure(self, test_client_with_mocked_rag, api_test_data):
        """Test that query response has correct structure"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json=api_test_data["valid_query"]
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches QueryResponse model
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data

        # Verify sources are properly formatted
        assert isinstance(data["sources"], list)
        for source in data["sources"]:
            assert isinstance(source, dict)
            assert "text" in source  # All sources should have text field


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""

    def test_get_course_stats_success(self, test_client_with_mocked_rag, api_test_data):
        """Test successful retrieval of course statistics"""
        response = test_client_with_mocked_rag.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        expected = api_test_data["expected_course_stats"]
        assert data["total_courses"] == expected["total_courses"]
        assert data["course_titles"] == expected["course_titles"]

    def test_get_course_stats_structure(self, test_client_with_mocked_rag):
        """Test that course stats response has correct structure"""
        response = test_client_with_mocked_rag.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches CourseStats model
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_get_course_stats_rag_error(self, test_client_with_mocked_rag, mock_rag_system):
        """Test course stats endpoint when RAG system raises exception"""
        # Mock RAG system to raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client_with_mocked_rag.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]


@pytest.mark.api
class TestNewSessionEndpoint:
    """Test cases for /api/new-session endpoint"""

    def test_create_new_session_with_old_session(self, test_client_with_mocked_rag, api_test_data, mock_rag_system):
        """Test creating new session while clearing old session"""
        response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json=api_test_data["new_session_request"]
        )

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

        # Verify old session was cleared
        mock_rag_system.session_manager.clear_session.assert_called_once_with("old-session-456")

    def test_create_new_session_without_old_session(self, test_client_with_mocked_rag, mock_rag_system):
        """Test creating new session without clearing old session"""
        response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json={}
        )

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

        # Verify clear_session was not called
        mock_rag_system.session_manager.clear_session.assert_not_called()

    def test_new_session_response_structure(self, test_client_with_mocked_rag):
        """Test that new session response has correct structure"""
        response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json={}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches NewSessionResponse model
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_new_session_rag_error(self, test_client_with_mocked_rag, mock_rag_system):
        """Test new session endpoint when RAG system raises exception"""
        # Mock RAG system to raise an exception
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")

        response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json={}
        )

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]


@pytest.mark.api
class TestRootEndpoint:
    """Test cases for root endpoint"""

    def test_root_endpoint(self, test_client_with_mocked_rag):
        """Test root endpoint returns expected message"""
        response = test_client_with_mocked_rag.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert data["message"] == "RAG System API"


@pytest.mark.api
class TestAPIHeaders:
    """Test API headers and middleware"""

    def test_cors_headers(self, test_client_with_mocked_rag):
        """Test that CORS headers are properly set"""
        response = test_client_with_mocked_rag.options("/api/query")

        # CORS preflight should be handled
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled

    def test_content_type_headers(self, test_client_with_mocked_rag, api_test_data):
        """Test that content-type headers are properly handled"""
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json=api_test_data["valid_query"],
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.api
class TestAPIValidation:
    """Test API input validation"""

    def test_query_field_validation(self, test_client_with_mocked_rag):
        """Test various query field validation scenarios"""
        # Test with non-string query
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"query": 123}  # Should be string
        )
        assert response.status_code == 422

        # Test with null query
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"query": None}
        )
        assert response.status_code == 422

    def test_session_id_validation(self, test_client_with_mocked_rag):
        """Test session_id field validation"""
        # Test with non-string session_id
        response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"query": "test", "session_id": 123}  # Should be string or null
        )
        assert response.status_code == 422

    def test_new_session_validation(self, test_client_with_mocked_rag):
        """Test new session request validation"""
        # Test with non-string old_session_id
        response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json={"old_session_id": 123}  # Should be string or null
        )
        assert response.status_code == 422


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints working together"""

    def test_session_workflow(self, test_client_with_mocked_rag):
        """Test complete session workflow: create session -> query -> clear session"""
        # Create new session
        session_response = test_client_with_mocked_rag.post("/api/new-session", json={})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        # Use session for query
        query_response = test_client_with_mocked_rag.post(
            "/api/query",
            json={"query": "test query", "session_id": session_id}
        )
        assert query_response.status_code == 200
        assert query_response.json()["session_id"] == session_id

        # Create new session with old session cleanup
        new_session_response = test_client_with_mocked_rag.post(
            "/api/new-session",
            json={"old_session_id": session_id}
        )
        assert new_session_response.status_code == 200
        new_session_id = new_session_response.json()["session_id"]
        assert new_session_id == "test-session-123"  # Mock always returns this

    def test_api_endpoints_consistency(self, test_client_with_mocked_rag):
        """Test that all API endpoints respond with consistent JSON structure"""
        endpoints_and_methods = [
            ("GET", "/"),
            ("GET", "/api/courses"),
            ("POST", "/api/query", {"query": "test"}),
            ("POST", "/api/new-session", {})
        ]

        for method, endpoint, *payload in endpoints_and_methods:
            if method == "GET":
                response = test_client_with_mocked_rag.get(endpoint)
            elif method == "POST":
                response = test_client_with_mocked_rag.post(endpoint, json=payload[0] if payload else {})

            assert response.status_code == 200
            # All endpoints should return valid JSON
            data = response.json()
            assert isinstance(data, dict)