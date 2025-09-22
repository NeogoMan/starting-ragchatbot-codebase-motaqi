import os
import shutil
import sys
import tempfile
from typing import Generator, List
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore


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
        CHROMA_PATH="./test_chroma_db",
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
            Lesson(
                lesson_number=1,
                title="What is Machine Learning?",
                lesson_link="https://example.com/ml-course/lesson-1",
            ),
            Lesson(
                lesson_number=2,
                title="Supervised Learning",
                lesson_link="https://example.com/ml-course/lesson-2",
            ),
            Lesson(
                lesson_number=3,
                title="Unsupervised Learning",
                lesson_link="https://example.com/ml-course/lesson-3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in data without labeled examples.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
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
    mock_final_response.content = [
        Mock(text="Based on the search results, machine learning is...")
    ]

    mock_client.messages.create.side_effect = [mock_response, mock_final_response]

    return mock_client


@pytest.fixture
def ai_generator_with_mock(mock_anthropic_client):
    """AIGenerator with mocked Anthropic client"""
    with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
        mock_anthropic.return_value = mock_anthropic_client
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client
        return generator
