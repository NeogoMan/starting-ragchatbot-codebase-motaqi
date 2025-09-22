import os
from unittest.mock import patch

import anthropic
import chromadb
import pytest
from ai_generator import AIGenerator
from config import config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestSystemHealth:
    """System diagnostic tests to verify the actual system state and configuration"""

    def test_api_key_configuration(self):
        """Test that ANTHROPIC_API_KEY is properly configured"""
        api_key = config.ANTHROPIC_API_KEY

        assert api_key is not None, "ANTHROPIC_API_KEY is not set"
        assert api_key != "", "ANTHROPIC_API_KEY is empty"
        assert api_key != "your-api-key-here", "ANTHROPIC_API_KEY is still placeholder"

        # Check if it looks like a valid API key format
        if not api_key.startswith("test-"):  # Allow test keys in testing
            assert api_key.startswith(
                "sk-"
            ), f"API key should start with 'sk-', got: {api_key[:10]}..."

    def test_chroma_database_exists(self):
        """Test that ChromaDB database exists and is accessible"""
        chroma_path = config.CHROMA_PATH

        assert os.path.exists(
            chroma_path
        ), f"ChromaDB path does not exist: {chroma_path}"

        # Try to connect to the database
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            collections = client.list_collections()
            assert isinstance(collections, list), "Failed to list collections"
        except Exception as e:
            pytest.fail(f"Failed to connect to ChromaDB: {e}")

    def test_chroma_collections_exist(self):
        """Test that required ChromaDB collections exist"""
        try:
            client = chromadb.PersistentClient(path=config.CHROMA_PATH)
            collection_names = [col.name for col in client.list_collections()]

            assert (
                "course_catalog" in collection_names
            ), "course_catalog collection missing"
            assert (
                "course_content" in collection_names
            ), "course_content collection missing"
        except Exception as e:
            pytest.fail(f"Failed to check collections: {e}")

    def test_chroma_database_has_data(self):
        """Test that ChromaDB contains course data"""
        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )

            # Check course catalog
            course_count = store.get_course_count()
            assert (
                course_count > 0
            ), f"No courses found in database. Expected > 0, got {course_count}"

            # Check course titles
            course_titles = store.get_existing_course_titles()
            assert len(course_titles) > 0, "No course titles found"
            assert all(
                isinstance(title, str) for title in course_titles
            ), "Invalid course title format"

            print(f"✓ Found {course_count} courses: {course_titles}")

        except Exception as e:
            pytest.fail(f"Failed to check database data: {e}")

    def test_course_content_searchable(self):
        """Test that course content is searchable"""
        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )

            # Try a simple search
            results = store.search("learning")

            if results.error:
                pytest.fail(f"Search returned error: {results.error}")

            if results.is_empty():
                pytest.fail(
                    "Search returned no results - database may be empty or corrupted"
                )

            assert len(results.documents) > 0, "No documents in search results"
            assert len(results.metadata) > 0, "No metadata in search results"
            assert len(results.documents) == len(
                results.metadata
            ), "Documents and metadata count mismatch"

            print(f"✓ Search successful, found {len(results.documents)} results")

        except Exception as e:
            pytest.fail(f"Search functionality failed: {e}")

    def test_embedding_model_accessible(self):
        """Test that the embedding model can be loaded"""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.EMBEDDING_MODEL)

            # Test encoding
            test_text = "This is a test sentence"
            embedding = model.encode([test_text])

            assert embedding is not None, "Failed to generate embedding"
            assert len(embedding) > 0, "Empty embedding generated"
            assert len(embedding[0]) > 0, "Invalid embedding dimension"

            print(
                f"✓ Embedding model '{config.EMBEDDING_MODEL}' working, dimension: {len(embedding[0])}"
            )

        except Exception as e:
            pytest.fail(f"Embedding model failed: {e}")

    def test_anthropic_api_connectivity(self):
        """Test connectivity to Anthropic API (without making actual calls)"""
        try:
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

            # Just test that the client can be created
            assert client is not None, "Failed to create Anthropic client"

            print("✓ Anthropic client created successfully")

        except Exception as e:
            pytest.fail(f"Failed to create Anthropic client: {e}")

    def test_documents_folder_exists(self):
        """Test that the documents folder exists"""
        docs_path = "../docs"

        assert os.path.exists(
            docs_path
        ), f"Documents folder does not exist: {docs_path}"

        # Check for course files
        files = os.listdir(docs_path)
        course_files = [
            f for f in files if f.lower().endswith((".txt", ".pdf", ".docx"))
        ]

        assert len(course_files) > 0, f"No course files found in {docs_path}"

        print(f"✓ Found {len(course_files)} course files: {course_files}")

    def test_rag_system_initialization(self):
        """Test that RAGSystem can be initialized with current config"""
        try:
            rag_system = RAGSystem(config)

            assert rag_system.vector_store is not None, "VectorStore not initialized"
            assert rag_system.ai_generator is not None, "AIGenerator not initialized"
            assert rag_system.tool_manager is not None, "ToolManager not initialized"
            assert (
                rag_system.search_tool is not None
            ), "CourseSearchTool not initialized"
            assert (
                rag_system.outline_tool is not None
            ), "CourseOutlineTool not initialized"

            # Test tool registration
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert (
                len(tool_definitions) == 2
            ), f"Expected 2 tools, got {len(tool_definitions)}"

            tool_names = [tool["name"] for tool in tool_definitions]
            assert (
                "search_course_content" in tool_names
            ), "search_course_content tool not registered"
            assert (
                "get_course_outline" in tool_names
            ), "get_course_outline tool not registered"

            print("✓ RAGSystem initialized successfully")

        except Exception as e:
            pytest.fail(f"RAGSystem initialization failed: {e}")

    def test_course_search_tool_functionality(self):
        """Test that CourseSearchTool works with actual data"""
        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )
            from search_tools import CourseSearchTool

            search_tool = CourseSearchTool(store)

            # Test basic search
            result = search_tool.execute("machine learning")

            assert isinstance(result, str), "Search tool should return string"
            assert len(result) > 0, "Search tool returned empty result"

            if "No relevant content found" in result:
                pytest.fail("Search tool found no content - database may be empty")

            # Should contain course context
            assert (
                "[" in result and "]" in result
            ), "Result should contain course context headers"

            print("✓ CourseSearchTool working with actual data")

        except Exception as e:
            pytest.fail(f"CourseSearchTool functionality test failed: {e}")

    def test_course_outline_tool_functionality(self):
        """Test that CourseOutlineTool works with actual data"""
        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )
            from search_tools import CourseOutlineTool

            outline_tool = CourseOutlineTool(store)

            # Get list of available courses first
            course_titles = store.get_existing_course_titles()
            if not course_titles:
                pytest.fail("No courses available to test outline tool")

            # Test with first available course
            first_course = course_titles[0]
            result = outline_tool.execute(first_course)

            assert isinstance(result, str), "Outline tool should return string"
            assert len(result) > 0, "Outline tool returned empty result"

            if "No course found matching" in result:
                pytest.fail(f"Outline tool couldn't find course: {first_course}")

            # Should contain course information
            assert "Course Title:" in result, "Result should contain course title"
            assert "Course Link:" in result, "Result should contain course link"

            print(f"✓ CourseOutlineTool working for course: {first_course}")

        except Exception as e:
            pytest.fail(f"CourseOutlineTool functionality test failed: {e}")

    def test_session_manager_functionality(self):
        """Test that SessionManager works correctly"""
        try:
            from session_manager import SessionManager

            session_manager = SessionManager(config.MAX_HISTORY)

            # Test session creation
            session_id = session_manager.create_session()
            assert isinstance(session_id, str), "Session ID should be string"
            assert len(session_id) > 0, "Session ID should not be empty"

            # Test adding exchanges
            session_manager.add_exchange(session_id, "Test query", "Test response")

            # Test getting history
            history = session_manager.get_conversation_history(session_id)
            assert isinstance(history, str), "History should be string"
            assert "Test query" in history, "History should contain query"
            assert "Test response" in history, "History should contain response"

            print(f"✓ SessionManager working, session: {session_id}")

        except Exception as e:
            pytest.fail(f"SessionManager functionality test failed: {e}")

    def test_system_configuration_values(self):
        """Test that system configuration values are reasonable"""
        # Check chunk sizes
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_SIZE <= 2000, "CHUNK_SIZE seems too large"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert (
            config.CHUNK_OVERLAP < config.CHUNK_SIZE
        ), "CHUNK_OVERLAP must be less than CHUNK_SIZE"

        # Check result limits
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be positive"
        assert config.MAX_RESULTS <= 20, "MAX_RESULTS seems too large"

        # Check history limit
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be non-negative"
        assert config.MAX_HISTORY <= 10, "MAX_HISTORY seems too large"

        # Check model names
        assert config.ANTHROPIC_MODEL.startswith(
            "claude"
        ), "ANTHROPIC_MODEL should be a Claude model"
        assert config.EMBEDDING_MODEL != "", "EMBEDDING_MODEL should not be empty"

        print(f"✓ Configuration values are reasonable")

    def test_actual_search_quality(self):
        """Test that search returns relevant results for common queries"""
        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )

            test_queries = [
                "machine learning",
                "artificial intelligence",
                "neural networks",
                "data science",
                "programming",
            ]

            successful_queries = 0

            for query in test_queries:
                results = store.search(query)

                if not results.error and not results.is_empty():
                    successful_queries += 1
                    print(f"✓ Query '{query}' found {len(results.documents)} results")

            # At least half the queries should return results
            assert (
                successful_queries >= len(test_queries) / 2
            ), f"Only {successful_queries}/{len(test_queries)} queries returned results"

            print(
                f"✓ Search quality test passed: {successful_queries}/{len(test_queries)} queries successful"
            )

        except Exception as e:
            pytest.fail(f"Search quality test failed: {e}")

    def test_memory_usage(self):
        """Test that the system doesn't use excessive memory"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Initialize system components
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )
            rag_system = RAGSystem(config)

            # Do some operations
            store.search("test query")
            rag_system.get_course_analytics()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 500MB)
            assert (
                memory_increase < 500
            ), f"Memory usage increased by {memory_increase:.1f}MB - too high"

            print(f"✓ Memory usage test passed: {memory_increase:.1f}MB increase")

        except Exception as e:
            pytest.fail(f"Memory usage test failed: {e}")

    @pytest.mark.slow
    def test_performance_benchmarks(self):
        """Test that search performance is reasonable"""
        import time

        try:
            store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )

            # Warm up
            store.search("warmup query")

            # Benchmark search
            start_time = time.time()
            for i in range(10):
                store.search(f"test query {i}")
            end_time = time.time()

            avg_time = (end_time - start_time) / 10

            # Average search should be under 2 seconds
            assert avg_time < 2.0, f"Average search time {avg_time:.2f}s is too slow"

            print(f"✓ Performance test passed: {avg_time:.3f}s average search time")

        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")
