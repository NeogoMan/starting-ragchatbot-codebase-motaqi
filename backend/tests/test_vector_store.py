import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStore:
    """Unit tests for VectorStore functionality"""

    def test_vector_store_initialization(self, temp_chroma_db):
        """Test VectorStore initializes correctly"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=10)

        assert store.max_results == 10
        assert store.client is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, empty_vector_store, sample_course):
        """Test adding course metadata to vector store"""
        empty_vector_store.add_course_metadata(sample_course)

        # Verify course was added by checking existing titles
        titles = empty_vector_store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_add_course_content(self, empty_vector_store, sample_course_chunks):
        """Test adding course content chunks to vector store"""
        empty_vector_store.add_course_content(sample_course_chunks)

        # Try to search for content to verify it was added
        results = empty_vector_store.search("machine learning")
        assert not results.is_empty()

    def test_search_basic_functionality(self, populated_vector_store):
        """Test basic search functionality"""
        results = populated_vector_store.search("machine learning")

        assert isinstance(results, SearchResults)
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) > 0
        assert len(results.distances) > 0

    def test_search_with_course_filter(self, populated_vector_store):
        """Test search with course name filter"""
        results = populated_vector_store.search(
            "learning", course_name="Introduction to Machine Learning"
        )

        assert isinstance(results, SearchResults)
        if not results.is_empty():
            # All results should be from the specified course
            for meta in results.metadata:
                assert meta["course_title"] == "Introduction to Machine Learning"

    def test_search_with_lesson_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("learning", lesson_number=2)

        assert isinstance(results, SearchResults)
        if not results.is_empty():
            # All results should be from lesson 2
            for meta in results.metadata:
                assert meta["lesson_number"] == 2

    def test_search_with_both_filters(self, populated_vector_store):
        """Test search with both course and lesson filters"""
        results = populated_vector_store.search(
            "supervised",
            course_name="Introduction to Machine Learning",
            lesson_number=2,
        )

        assert isinstance(results, SearchResults)
        if not results.is_empty():
            for meta in results.metadata:
                assert meta["course_title"] == "Introduction to Machine Learning"
                assert meta["lesson_number"] == 2

    def test_search_nonexistent_course(self, populated_vector_store):
        """Test search with nonexistent course name"""
        results = populated_vector_store.search(
            "test", course_name="Nonexistent Course"
        )

        assert isinstance(results, SearchResults)
        assert results.error is not None
        assert "No course found matching" in results.error

    def test_resolve_course_name_exact_match(self, populated_vector_store):
        """Test course name resolution with exact match"""
        resolved = populated_vector_store._resolve_course_name(
            "Introduction to Machine Learning"
        )
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_course_name_partial_match(self, populated_vector_store):
        """Test course name resolution with partial match"""
        resolved = populated_vector_store._resolve_course_name("Machine Learning")
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_course_name_case_insensitive(self, populated_vector_store):
        """Test course name resolution is case insensitive"""
        resolved = populated_vector_store._resolve_course_name("machine learning")
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_course_name_not_found(self, populated_vector_store):
        """Test course name resolution when course doesn't exist"""
        resolved = populated_vector_store._resolve_course_name("Quantum Computing")
        assert resolved is None

    def test_build_filter_no_filters(self, populated_vector_store):
        """Test filter building with no filters"""
        filter_dict = populated_vector_store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, populated_vector_store):
        """Test filter building with course filter only"""
        filter_dict = populated_vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, populated_vector_store):
        """Test filter building with lesson filter only"""
        filter_dict = populated_vector_store._build_filter(None, 2)
        assert filter_dict == {"lesson_number": 2}

    def test_build_filter_both_filters(self, populated_vector_store):
        """Test filter building with both filters"""
        filter_dict = populated_vector_store._build_filter("Test Course", 2)
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 2}]}
        assert filter_dict == expected

    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()
        assert isinstance(titles, list)
        assert "Introduction to Machine Learning" in titles

    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_get_course_link(self, populated_vector_store):
        """Test getting course link"""
        link = populated_vector_store.get_course_link(
            "Introduction to Machine Learning"
        )
        assert link == "https://example.com/ml-course"

    def test_get_lesson_link(self, populated_vector_store):
        """Test getting lesson link"""
        link = populated_vector_store.get_lesson_link(
            "Introduction to Machine Learning", 1
        )
        assert link == "https://example.com/ml-course/lesson-1"

    def test_get_lesson_link_not_found(self, populated_vector_store):
        """Test getting lesson link when lesson doesn't exist"""
        link = populated_vector_store.get_lesson_link(
            "Introduction to Machine Learning", 999
        )
        assert link is None

    def test_clear_all_data(self, populated_vector_store):
        """Test clearing all data from vector store"""
        # Verify data exists first
        assert populated_vector_store.get_course_count() > 0

        # Clear data
        populated_vector_store.clear_all_data()

        # Verify data is cleared
        assert populated_vector_store.get_course_count() == 0

    def test_search_with_limit(self, populated_vector_store):
        """Test search respects limit parameter"""
        results = populated_vector_store.search("learning", limit=1)

        assert isinstance(results, SearchResults)
        if not results.is_empty():
            assert len(results.documents) <= 1

    def test_search_with_custom_limit_vs_default(self, populated_vector_store):
        """Test search with custom limit vs default"""
        # Test with custom limit
        results_limited = populated_vector_store.search("learning", limit=1)

        # Test with default limit
        results_default = populated_vector_store.search("learning")

        # Custom limit should return fewer or equal results
        if not results_limited.is_empty() and not results_default.is_empty():
            assert len(results_limited.documents) <= len(results_default.documents)

    def test_search_results_from_chroma(self):
        """Test SearchResults.from_chroma() class method"""
        mock_chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [["meta1", "meta2"]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(mock_chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == ["meta1", "meta2"]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_search_results_empty(self):
        """Test SearchResults.empty() class method"""
        error_msg = "Test error"
        results = SearchResults.empty(error_msg)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
        assert results.is_empty()

    def test_search_results_is_empty(self):
        """Test SearchResults.is_empty() method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()

        # Non-empty results
        non_empty_results = SearchResults(["doc"], ["meta"], [0.1])
        assert not non_empty_results.is_empty()

    def test_get_all_courses_metadata(self, populated_vector_store):
        """Test getting all courses metadata"""
        metadata_list = populated_vector_store.get_all_courses_metadata()

        assert isinstance(metadata_list, list)
        assert len(metadata_list) > 0

        # Check structure of first course metadata
        course_meta = metadata_list[0]
        assert "title" in course_meta
        assert "instructor" in course_meta
        assert "course_link" in course_meta
        assert "lessons" in course_meta  # Should be parsed from JSON

    def test_add_empty_course_content(self, empty_vector_store):
        """Test adding empty course content list"""
        # Should not raise an error
        empty_vector_store.add_course_content([])

        # Should still be empty
        results = empty_vector_store.search("anything")
        assert results.is_empty()

    @patch("vector_store.chromadb.PersistentClient")
    def test_search_database_error(self, mock_client_class, temp_chroma_db):
        """Test search when database throws an error"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database error")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        results = store.search("test query")

        assert isinstance(results, SearchResults)
        assert results.error is not None
        assert "Search error" in results.error

    def test_course_metadata_serialization(self, empty_vector_store, sample_course):
        """Test that course metadata with lessons is properly serialized"""
        empty_vector_store.add_course_metadata(sample_course)

        # Get the metadata back
        metadata_list = empty_vector_store.get_all_courses_metadata()
        assert len(metadata_list) == 1

        course_meta = metadata_list[0]
        assert course_meta["title"] == sample_course.title
        assert course_meta["instructor"] == sample_course.instructor
        assert len(course_meta["lessons"]) == len(sample_course.lessons)

        # Check lesson structure
        lesson = course_meta["lessons"][0]
        assert "lesson_number" in lesson
        assert "lesson_title" in lesson
        assert "lesson_link" in lesson

    @pytest.mark.parametrize("max_results", [1, 3, 5, 10])
    def test_search_respects_max_results(
        self, temp_chroma_db, sample_course, sample_course_chunks, max_results
    ):
        """Test that search respects max_results configuration"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=max_results)

        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search
        results = store.search("learning")

        # Should not exceed max_results
        if not results.is_empty():
            assert len(results.documents) <= max_results
