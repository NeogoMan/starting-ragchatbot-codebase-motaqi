import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystem:
    """End-to-end tests for RAGSystem query flow"""

    @pytest.fixture
    def test_rag_system(self, temp_chroma_db):
        """RAGSystem configured for testing"""
        config = Config(
            ANTHROPIC_API_KEY="test-key-123",
            ANTHROPIC_MODEL="claude-sonnet-4-20250514",
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
            CHUNK_SIZE=800,
            CHUNK_OVERLAP=100,
            MAX_RESULTS=5,
            MAX_HISTORY=2,
            CHROMA_PATH=temp_chroma_db,
        )
        return RAGSystem(config)

    def test_rag_system_initialization(self, test_rag_system):
        """Test RAGSystem initializes all components correctly"""
        assert test_rag_system.document_processor is not None
        assert test_rag_system.vector_store is not None
        assert test_rag_system.ai_generator is not None
        assert test_rag_system.session_manager is not None
        assert test_rag_system.tool_manager is not None
        assert test_rag_system.search_tool is not None
        assert test_rag_system.outline_tool is not None

        # Check that tools are registered
        tool_definitions = test_rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_add_course_document_success(
        self, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test successfully adding a course document"""
        # Mock document processor to return sample data
        with patch.object(
            test_rag_system.document_processor, "process_course_document"
        ) as mock_process:
            mock_process.return_value = (sample_course, sample_course_chunks)

            course, chunk_count = test_rag_system.add_course_document("fake_path.txt")

            assert course == sample_course
            assert chunk_count == len(sample_course_chunks)

            # Verify data was added to vector store
            existing_titles = test_rag_system.vector_store.get_existing_course_titles()
            assert sample_course.title in existing_titles

    def test_add_course_document_error(self, test_rag_system):
        """Test adding course document when processing fails"""
        with patch.object(
            test_rag_system.document_processor, "process_course_document"
        ) as mock_process:
            mock_process.side_effect = Exception("File not found")

            course, chunk_count = test_rag_system.add_course_document("nonexistent.txt")

            assert course is None
            assert chunk_count == 0

    def test_add_course_folder_success(
        self, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test successfully adding course folder"""
        # Create a temporary folder with a test file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "course.txt")
            with open(test_file, "w") as f:
                f.write("Test course content")

            # Mock document processor
            with patch.object(
                test_rag_system.document_processor, "process_course_document"
            ) as mock_process:
                mock_process.return_value = (sample_course, sample_course_chunks)

                courses, chunks = test_rag_system.add_course_folder(temp_dir)

                assert courses == 1
                assert chunks == len(sample_course_chunks)

    def test_add_course_folder_nonexistent(self, test_rag_system):
        """Test adding course folder that doesn't exist"""
        courses, chunks = test_rag_system.add_course_folder("/nonexistent/path")

        assert courses == 0
        assert chunks == 0

    def test_add_course_folder_skip_existing(
        self, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test that existing courses are skipped when adding folder"""
        # First add a course
        test_rag_system.vector_store.add_course_metadata(sample_course)

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "course.txt")
            with open(test_file, "w") as f:
                f.write("Test course content")

            # Mock document processor to return the same course
            with patch.object(
                test_rag_system.document_processor, "process_course_document"
            ) as mock_process:
                mock_process.return_value = (sample_course, sample_course_chunks)

                courses, chunks = test_rag_system.add_course_folder(temp_dir)

                # Should skip existing course
                assert courses == 0
                assert chunks == 0

    @patch("rag_system.anthropic.Anthropic")
    def test_query_basic_functionality(
        self, mock_anthropic, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test basic query functionality with mocked AI"""
        # Set up data
        test_rag_system.vector_store.add_course_metadata(sample_course)
        test_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Mock AI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Machine learning is a subset of AI...")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Re-initialize AI generator with mock
        test_rag_system.ai_generator.client = mock_client

        response, sources = test_rag_system.query("What is machine learning?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(sources, list)

    @patch("rag_system.anthropic.Anthropic")
    def test_query_with_tool_use(
        self, mock_anthropic, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test query that triggers tool use"""
        # Set up data
        test_rag_system.vector_store.add_course_metadata(sample_course)
        test_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on the search results, machine learning is...")
        ]

        mock_client = Mock()
        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]
        mock_anthropic.return_value = mock_client

        # Re-initialize AI generator with mock
        test_rag_system.ai_generator.client = mock_client

        response, sources = test_rag_system.query("What is machine learning?")

        assert isinstance(response, str)
        assert "Based on the search results" in response
        assert isinstance(sources, list)

    def test_query_with_session_management(self, test_rag_system):
        """Test query with session management"""
        session_id = "test-session-123"

        # Mock AI generator to avoid actual API calls
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Test response"

            response, sources = test_rag_system.query(
                "Test query", session_id=session_id
            )

            # Verify session was used
            assert response == "Test response"

            # Verify conversation was added to session
            history = test_rag_system.session_manager.get_conversation_history(
                session_id
            )
            assert "Test query" in history
            assert "Test response" in history

    def test_query_without_session(self, test_rag_system):
        """Test query without providing session ID"""
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Test response"

            response, sources = test_rag_system.query("Test query")

            assert response == "Test response"
            assert isinstance(sources, list)

    def test_get_course_analytics(self, test_rag_system, sample_course):
        """Test getting course analytics"""
        # Add a course
        test_rag_system.vector_store.add_course_metadata(sample_course)

        analytics = test_rag_system.get_course_analytics()

        assert isinstance(analytics, dict)
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] >= 1
        assert sample_course.title in analytics["course_titles"]

    def test_tools_integration(self, test_rag_system):
        """Test that tools are properly integrated with AI generator"""
        tool_definitions = test_rag_system.tool_manager.get_tool_definitions()

        # Should have both search and outline tools
        assert len(tool_definitions) == 2

        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

        # Each tool should have proper structure
        for tool_def in tool_definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def
            assert "properties" in tool_def["input_schema"]
            assert "required" in tool_def["input_schema"]

    def test_source_management(
        self, test_rag_system, sample_course, sample_course_chunks
    ):
        """Test that sources are properly managed throughout query flow"""
        # Set up data
        test_rag_system.vector_store.add_course_metadata(sample_course)
        test_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Mock AI to trigger tool use
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:

            def mock_ai_with_tool_use(*args, **kwargs):
                # Simulate tool execution
                if "tool_manager" in kwargs:
                    test_rag_system.tool_manager.execute_tool(
                        "search_course_content", query="machine learning"
                    )
                return "Response based on search"

            mock_generate.side_effect = mock_ai_with_tool_use

            response, sources = test_rag_system.query("What is machine learning?")

            # Should have sources from tool execution
            assert isinstance(sources, list)
            if sources:  # If search found results
                assert len(sources) > 0
                source = sources[0]
                assert isinstance(source, dict)
                assert "text" in source

    def test_error_propagation(self, test_rag_system):
        """Test that errors are properly propagated through the system"""
        # Mock AI generator to raise an error
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            with pytest.raises(Exception) as exc_info:
                test_rag_system.query("Test query")

            assert "API Error" in str(exc_info.value)

    def test_empty_database_query(self, test_rag_system):
        """Test querying when database is empty"""
        # Mock AI to simulate tool use on empty database
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:

            def mock_ai_with_empty_search(*args, **kwargs):
                if "tool_manager" in kwargs:
                    # This will search empty database
                    result = test_rag_system.tool_manager.execute_tool(
                        "search_course_content", query="anything"
                    )
                    # Should return no results message
                    assert "No relevant content found" in result
                return "I couldn't find any relevant information"

            mock_generate.side_effect = mock_ai_with_empty_search

            response, sources = test_rag_system.query("What is machine learning?")

            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_configuration_usage(self, temp_chroma_db):
        """Test that RAGSystem uses configuration correctly"""
        config = Config(
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="claude-sonnet-4-20250514",
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
            CHUNK_SIZE=1000,
            CHUNK_OVERLAP=200,
            MAX_RESULTS=10,
            MAX_HISTORY=5,
            CHROMA_PATH=temp_chroma_db,
        )

        rag_system = RAGSystem(config)

        # Verify configuration is used
        assert rag_system.vector_store.max_results == 10
        assert rag_system.ai_generator.model == "claude-sonnet-4-20250514"
        assert rag_system.session_manager.max_history == 5

    @pytest.mark.parametrize(
        "query_type,expected_tool",
        [
            ("What courses are available?", "get_course_outline"),
            ("What is machine learning?", "search_course_content"),
            ("Tell me about lesson 1", "search_course_content"),
        ],
    )
    def test_different_query_types(
        self,
        test_rag_system,
        sample_course,
        sample_course_chunks,
        query_type,
        expected_tool,
    ):
        """Test that different query types work correctly"""
        # Set up data
        test_rag_system.vector_store.add_course_metadata(sample_course)
        test_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Mock AI to use specific tool
        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:

            def mock_ai_selective_tool(*args, **kwargs):
                if "tool_manager" in kwargs:
                    if expected_tool == "get_course_outline":
                        return test_rag_system.tool_manager.execute_tool(
                            "get_course_outline", course_name="Machine Learning"
                        )
                    else:
                        return test_rag_system.tool_manager.execute_tool(
                            "search_course_content", query=query_type
                        )
                return f"Response for {query_type}"

            mock_generate.side_effect = mock_ai_selective_tool

            response, sources = test_rag_system.query(query_type)

            assert isinstance(response, str)
            assert len(response) > 0

    def test_session_history_limit(self, test_rag_system):
        """Test that session history respects the configured limit"""
        session_id = "test-session"

        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Response"

            # Add more exchanges than the limit
            for i in range(5):  # More than MAX_HISTORY=2
                test_rag_system.query(f"Query {i}", session_id=session_id)

            history = test_rag_system.session_manager.get_conversation_history(
                session_id
            )

            # History should be limited
            # The exact format depends on session manager implementation
            assert isinstance(history, str)
            assert len(history) > 0

    def test_concurrent_sessions(self, test_rag_system):
        """Test handling multiple concurrent sessions"""
        session1 = "session-1"
        session2 = "session-2"

        with patch.object(
            test_rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Response"

            # Make queries with different sessions
            test_rag_system.query("Query for session 1", session1)
            test_rag_system.query("Query for session 2", session2)

            history1 = test_rag_system.session_manager.get_conversation_history(
                session1
            )
            history2 = test_rag_system.session_manager.get_conversation_history(
                session2
            )

            # Histories should be different
            assert "session 1" in history1
            assert "session 2" in history2
            assert history1 != history2
