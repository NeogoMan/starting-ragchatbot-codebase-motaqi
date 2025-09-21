import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestCourseSearchTool:
    """Unit tests for CourseSearchTool.execute() method"""

    def test_tool_definition(self, course_search_tool):
        """Test that tool definition is properly formatted"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]

        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

    def test_execute_basic_search_success(self, course_search_tool):
        """Test successful execution with basic query"""
        result = course_search_tool.execute("machine learning")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain course context
        assert "Introduction to Machine Learning" in result
        # Should contain content
        assert "artificial intelligence" in result or "subset" in result

    def test_execute_with_course_filter(self, course_search_tool):
        """Test execution with course name filter"""
        result = course_search_tool.execute("learning", course_name="Introduction")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to Machine Learning" in result

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test execution with lesson number filter"""
        result = course_search_tool.execute("supervised", lesson_number=2)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Lesson 2" in result
        assert "supervised" in result.lower()

    def test_execute_with_both_filters(self, course_search_tool):
        """Test execution with both course and lesson filters"""
        result = course_search_tool.execute(
            "supervised learning",
            course_name="Machine Learning",
            lesson_number=2
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to Machine Learning" in result
        assert "Lesson 2" in result

    def test_execute_no_results(self, course_search_tool):
        """Test execution when no results are found"""
        result = course_search_tool.execute("quantum computing")

        assert isinstance(result, str)
        assert "No relevant content found" in result

    def test_execute_no_results_with_filters(self, course_search_tool):
        """Test execution with filters when no results are found"""
        result = course_search_tool.execute(
            "machine learning",
            course_name="Nonexistent Course"
        )

        assert isinstance(result, str)
        assert "No course found matching" in result

    def test_execute_invalid_course_name(self, course_search_tool):
        """Test execution with invalid course name"""
        result = course_search_tool.execute("test", course_name="Invalid Course")

        assert isinstance(result, str)
        assert "No course found matching" in result

    def test_execute_stores_sources(self, course_search_tool):
        """Test that execution stores sources for retrieval"""
        # Clear any existing sources
        course_search_tool.last_sources = []

        result = course_search_tool.execute("machine learning")

        # Should have stored sources
        assert len(course_search_tool.last_sources) > 0
        source = course_search_tool.last_sources[0]
        assert "text" in source
        assert "Introduction to Machine Learning" in source["text"]

    def test_execute_with_vector_store_error(self, empty_vector_store):
        """Test execution when vector store returns error"""
        tool = CourseSearchTool(empty_vector_store)

        # Mock the vector store to return an error
        with patch.object(empty_vector_store, 'search') as mock_search:
            mock_search.return_value = SearchResults.empty("Database connection failed")

            result = tool.execute("test query")

            assert isinstance(result, str)
            assert "Database connection failed" in result

    def test_execute_result_formatting(self, course_search_tool):
        """Test that results are properly formatted"""
        result = course_search_tool.execute("machine learning")

        # Should contain course context header
        assert "[Introduction to Machine Learning" in result

        # Should have lesson information if applicable
        lines = result.split('\n')
        assert len(lines) > 1  # Should have multiple lines (header + content)

    @pytest.mark.parametrize("query,expected_in_result", [
        ("machine learning", "machine learning"),
        ("supervised", "supervised"),
        ("artificial intelligence", "artificial"),
    ])
    def test_execute_various_queries(self, course_search_tool, query, expected_in_result):
        """Test execution with various query types"""
        result = course_search_tool.execute(query)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain relevant content (case insensitive)
        assert expected_in_result.lower() in result.lower()

    def test_execute_empty_query(self, course_search_tool):
        """Test execution with empty query"""
        result = course_search_tool.execute("")

        # Should still work, might return generic results or error
        assert isinstance(result, str)

    def test_sources_reset_behavior(self, course_search_tool):
        """Test that sources are properly managed"""
        # Execute first search
        course_search_tool.execute("machine learning")
        first_sources = course_search_tool.last_sources.copy()

        # Execute second search
        course_search_tool.execute("supervised learning")
        second_sources = course_search_tool.last_sources

        # Sources should be updated, not cumulative
        assert len(second_sources) > 0
        # Sources should be different (assuming different results)
        assert first_sources != second_sources

    def test_execute_with_limit_results(self, populated_vector_store):
        """Test execution respects result limits"""
        # Create tool with populated store
        tool = CourseSearchTool(populated_vector_store)

        # Execute search - should respect the vector store's max_results
        result = tool.execute("learning")

        # Should have results but not infinite
        assert isinstance(result, str)
        assert len(result) > 0

        # Check that sources are limited
        assert len(tool.last_sources) <= populated_vector_store.max_results

    def test_format_results_with_lesson_links(self, course_search_tool):
        """Test that lesson links are included in sources when available"""
        result = course_search_tool.execute("machine learning", lesson_number=1)

        if course_search_tool.last_sources:
            source = course_search_tool.last_sources[0]
            # Should have text
            assert "text" in source
            # May have URL if lesson link is available
            if "url" in source:
                assert source["url"].startswith("http")

    def test_tool_interface_compliance(self, course_search_tool):
        """Test that CourseSearchTool properly implements Tool interface"""
        # Should have required methods
        assert hasattr(course_search_tool, 'get_tool_definition')
        assert hasattr(course_search_tool, 'execute')

        # Methods should be callable
        assert callable(course_search_tool.get_tool_definition)
        assert callable(course_search_tool.execute)

        # get_tool_definition should return dict
        definition = course_search_tool.get_tool_definition()
        assert isinstance(definition, dict)

        # execute should accept keyword arguments
        result = course_search_tool.execute(query="test")
        assert isinstance(result, str)