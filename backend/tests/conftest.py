import pytest
import tempfile
import shutil
import os
from typing import Generator, List
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from fastapi import FastAPI
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with temporary paths"""
    return Config(
        ANTHROPIC_API_KEY="test-key-123",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db"
    )


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is Machine Learning?", lesson_link="https://example.com/ml-course/lesson-1"),
            Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml-course/lesson-2"),
            Lesson(lesson_number=3, title="Unsupervised Learning", lesson_link="https://example.com/ml-course/lesson-3"),
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in data without labeled examples.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        ),
    ]


@pytest.fixture
def populated_vector_store(temp_chroma_db, sample_course, sample_course_chunks):
    """Vector store populated with test data"""
    config = Config(CHROMA_PATH=temp_chroma_db)
    store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

    # Add course metadata and content
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_course_chunks)

    return store


@pytest.fixture
def empty_vector_store(temp_chroma_db):
    """Empty vector store for testing"""
    config = Config(CHROMA_PATH=temp_chroma_db)
    return VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)


@pytest.fixture
def course_search_tool(populated_vector_store):
    """CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)


@pytest.fixture
def course_outline_tool(populated_vector_store):
    """CourseOutlineTool with populated data"""
    return CourseOutlineTool(populated_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock a successful text response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic client that returns tool use response"""
    mock_client = Mock()

    # Mock tool use response
    mock_content_block = Mock()
    mock_content_block.type = "tool_use"
    mock_content_block.name = "search_course_content"
    mock_content_block.id = "tool_123"
    mock_content_block.input = {"query": "machine learning"}

    mock_response = Mock()
    mock_response.content = [mock_content_block]
    mock_response.stop_reason = "tool_use"

    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock(text="Based on the search results, machine learning is...")]

    mock_client.messages.create.side_effect = [mock_response, mock_final_response]

    return mock_client


@pytest.fixture
def ai_generator_with_mock(mock_anthropic_client):
    """AIGenerator with mocked Anthropic client"""
    with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = mock_anthropic_client
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client
        return generator


# API Testing Fixtures

@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any

    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Define models inline to avoid import issues
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class NewSessionRequest(BaseModel):
        old_session_id: Optional[str] = None

    class NewSessionResponse(BaseModel):
        session_id: str

    return app, QueryRequest, QueryResponse, CourseStats, NewSessionRequest, NewSessionResponse


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test answer about machine learning.",
        [{"text": "Machine learning is a subset of AI", "url": "https://example.com/ml"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to ML", "Advanced Python"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.clear_session.return_value = None
    return mock_rag


@pytest.fixture
def test_client_with_mocked_rag(test_app, mock_rag_system):
    """Test client with mocked RAG system"""
    from fastapi import HTTPException

    app, QueryRequest, QueryResponse, CourseStats, NewSessionRequest, NewSessionResponse = test_app

    # Define endpoints with mock RAG system
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            formatted_sources = []
            for source in sources:
                if isinstance(source, dict):
                    formatted_sources.append(source)
                else:
                    formatted_sources.append({"text": str(source)})

            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/new-session", response_model=NewSessionResponse)
    async def create_new_session(request: NewSessionRequest):
        try:
            if request.old_session_id:
                mock_rag_system.session_manager.clear_session(request.old_session_id)

            new_session_id = mock_rag_system.session_manager.create_session()
            return NewSessionResponse(session_id=new_session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def read_root():
        return {"message": "RAG System API"}

    return TestClient(app)


@pytest.fixture
def api_test_data():
    """Common test data for API tests"""
    return {
        "valid_query": {
            "query": "What is machine learning?",
            "session_id": "test-session-123"
        },
        "query_without_session": {
            "query": "Explain supervised learning"
        },
        "invalid_query": {
            "query": "",  # Empty query
            "session_id": "test-session-123"
        },
        "new_session_request": {
            "old_session_id": "old-session-456"
        },
        "expected_course_stats": {
            "total_courses": 2,
            "course_titles": ["Introduction to ML", "Advanced Python"]
        },
        "expected_query_response": {
            "answer": "This is a test answer about machine learning.",
            "sources": [{"text": "Machine learning is a subset of AI", "url": "https://example.com/ml"}],
            "session_id": "test-session-123"
        }
    }