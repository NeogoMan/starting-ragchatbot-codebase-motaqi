# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- Quick start: `./run.sh` (after making executable with `chmod +x run.sh`)
- Manual start: `cd backend && uv run uvicorn app:app --reload --port 8000`
- Dependency management: `uv sync` to install/update dependencies

### Environment Setup
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- The application requires Python 3.13+ and uv package manager

### Code Quality Tools
- Format code: `./scripts/format.sh` (runs black and isort)
- Lint code: `./scripts/lint.sh` (runs flake8 and mypy)
- Complete quality check: `./scripts/quality-check.sh` (formats, lints, and tests)
- Individual commands:
  - `uv run black .` - Format code with black
  - `uv run isort .` - Sort imports
  - `uv run flake8 .` - Lint with flake8
  - `uv run mypy .` - Type check with mypy
  - `uv run pytest` - Run tests

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials with the following key components:

### Core System Architecture
- **RAGSystem** (`rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`vector_store.py`): ChromaDB-based vector storage for semantic search
- **DocumentProcessor** (`document_processor.py`): Handles document parsing and chunking
- **AIGenerator** (`ai_generator.py`): Anthropic Claude integration with tool support
- **SessionManager** (`session_manager.py`): Manages conversation context and history
- **ToolManager** (`search_tools.py`): Provides tool-based search capabilities for the AI

### Application Structure
- **Backend**: FastAPI application (`app.py`) serving both API endpoints and static frontend
  - Main endpoints: `/api/query` (process questions), `/api/courses` (get analytics)
  - Startup process automatically loads documents from `../docs` folder
- **Frontend**: Static HTML/CSS/JS served from `/frontend` directory
- **Data Flow**: Documents → Processing → Vector Storage → Query → AI Generation with Tools → Response

### Key Design Patterns
- Tool-based AI interaction: The AI uses structured tools to search the knowledge base
- Session-based conversations: Each user interaction maintains context through sessions
- Modular component design: Each major function is separated into its own class/module
- Lazy loading: Documents are processed and stored on first startup, then reused

### Data Models
- **Course**: Represents a complete course with title, description, and lessons
- **Lesson**: Individual sections within a course
- **CourseChunk**: Text chunks for vector storage with metadata

### Storage
- Vector database stored in `backend/chroma_db/`
- No traditional database - everything persisted in ChromaDB collections
- Course metadata and content chunks stored separately for optimized retrieval