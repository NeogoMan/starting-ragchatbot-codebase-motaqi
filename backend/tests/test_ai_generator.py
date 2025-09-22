from unittest.mock import MagicMock, Mock, patch

import anthropic
import pytest
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


class TestAIGenerator:
    """Integration tests for AIGenerator tool calling functionality"""

    def test_ai_generator_initialization(self):
        """Test AIGenerator initializes correctly"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, ai_generator_with_mock):
        """Test generating response without tools"""
        response = ai_generator_with_mock.generate_response("What is machine learning?")

        assert isinstance(response, str)
        assert len(response) > 0
        # Verify API was called
        ai_generator_with_mock.client.messages.create.assert_called_once()

    def test_generate_response_with_conversation_history(self, ai_generator_with_mock):
        """Test generating response with conversation history"""
        history = "Previous conversation about AI"
        response = ai_generator_with_mock.generate_response(
            "What is machine learning?", conversation_history=history
        )

        assert isinstance(response, str)
        # Verify history was included in system content
        call_args = ai_generator_with_mock.client.messages.create.call_args
        assert "Previous conversation" in call_args[1]["system"]

    def test_generate_response_with_tools_no_tool_use(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test generating response with tools available but not used"""
        # Mock response that doesn't use tools
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a general response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        response = ai_generator_with_mock.generate_response(
            "What is the weather?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert isinstance(response, str)
        assert response == "This is a general response"

    def test_generate_response_with_tool_use(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test generating response with tool use"""
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on the search results...")]

        ai_generator_with_mock.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        response = ai_generator_with_mock.generate_response(
            "What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert isinstance(response, str)
        assert "Based on the search results" in response
        # Verify two API calls were made (initial + final)
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_handle_sequential_tool_execution_single_round(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test _handle_sequential_tool_execution with single tool call round"""
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response (no tool use)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response based on search")]
        mock_final_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": tool_manager.get_tool_definitions(),
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_initial_response, base_params, tool_manager
        )

        assert result == "Final response based on search"

    def test_handle_sequential_tool_execution_two_rounds(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test _handle_sequential_tool_execution with two sequential tool call rounds"""
        # Mock Round 1: Course outline tool
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "get_course_outline"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"course_name": "MCP"}

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block1]
        mock_round1_response.stop_reason = "tool_use"

        # Mock Round 2: Search based on outline results
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"query": "servers configuration"}

        mock_round2_response = Mock()
        mock_round2_response.content = [mock_tool_block2]
        mock_round2_response.stop_reason = "tool_use"

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Based on the MCP course outline and search results, servers configuration..."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        # Set up sequence of API calls
        ai_generator_with_mock.client.messages.create.side_effect = [
            mock_round2_response,  # Response to round 1 tool results
            mock_final_response,  # Response to round 2 tool results
        ]

        base_params = {
            "messages": [
                {
                    "role": "user",
                    "content": "Find content about servers configuration in MCP course",
                }
            ],
            "system": "test system prompt",
            "tools": tool_manager.get_tool_definitions(),
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_round1_response,  # Initial response with first tool call
            base_params,
            tool_manager,
        )

        assert "servers configuration" in result
        # Verify two additional API calls were made (one for each round)
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_tool_execution_error_handling(self, ai_generator_with_mock):
        """Test tool execution when tool manager throws error"""
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "nonexistent_tool"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]

        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Tool 'nonexistent_tool' not found"
        )

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled response")]
        ai_generator_with_mock.client.messages.create.return_value = mock_final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
        }

        result = ai_generator_with_mock._handle_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        assert result == "Error handled response"

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_key_configuration(self, mock_anthropic):
        """Test that API key is properly passed to Anthropic client"""
        api_key = "test-api-key-123"
        generator = AIGenerator(api_key, "claude-sonnet-4-20250514")

        mock_anthropic.assert_called_once_with(api_key=api_key)

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            "Rate limit exceeded"
        )
        mock_anthropic.return_value = mock_client

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        with pytest.raises(anthropic.APIError):
            generator.generate_response("test query")

    def test_system_prompt_structure(self, ai_generator_with_mock):
        """Test that system prompt contains required instructions"""
        ai_generator_with_mock.generate_response("test query")

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]

        # Check for key instructions in system prompt
        assert "course materials" in system_content.lower()
        assert "tool" in system_content.lower()
        assert "search" in system_content.lower()

    def test_message_structure(self, ai_generator_with_mock, tool_manager):
        """Test that messages are properly structured for API"""
        query = "What is machine learning?"
        ai_generator_with_mock.generate_response(
            query, tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        messages = call_args[1]["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == query

    def test_tools_parameter_structure(self, ai_generator_with_mock, tool_manager):
        """Test that tools parameter is properly structured"""
        ai_generator_with_mock.generate_response(
            "test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        tools = call_args[1]["tools"]

        assert isinstance(tools, list)
        assert len(tools) > 0
        # Each tool should have required structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_tool_choice_parameter(self, ai_generator_with_mock, tool_manager):
        """Test that tool_choice parameter is set correctly"""
        ai_generator_with_mock.generate_response(
            "test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        tool_choice = call_args[1]["tool_choice"]

        assert tool_choice == {"type": "auto"}

    def test_conversation_history_integration(self, ai_generator_with_mock):
        """Test conversation history is properly integrated into system prompt"""
        history = "User: Hello\nAssistant: Hi there!"
        query = "What is AI?"

        ai_generator_with_mock.generate_response(query, conversation_history=history)

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]

        assert "Previous conversation:" in system_content
        assert history in system_content

    def test_no_conversation_history(self, ai_generator_with_mock):
        """Test system prompt when no conversation history is provided"""
        query = "What is AI?"

        ai_generator_with_mock.generate_response(query)

        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args[1]["system"]

        assert "Previous conversation:" not in system_content

    def test_tool_result_message_format(self, ai_generator_with_mock, tool_manager):
        """Test that tool results are properly formatted in messages"""
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]

        # Create a mock that captures the final API call
        def capture_final_call(*args, **kwargs):
            # This should be the final call with tool results
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final response")]
            return mock_final_response

        ai_generator_with_mock.client.messages.create.side_effect = [
            mock_initial_response,
            capture_final_call,
        ]

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
        }

        ai_generator_with_mock._handle_tool_execution(
            mock_initial_response, base_params, tool_manager
        )

        # Verify that the second call included tool results
        second_call_args = ai_generator_with_mock.client.messages.create.call_args
        messages = second_call_args[1]["messages"]

        # Should have user message, assistant message with tool use, and user message with tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool result message should have proper structure
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_123"

    def test_sequential_tool_execution_max_rounds_limit(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test that sequential tool execution respects 2-round maximum"""
        # Mock tool responses for 3 potential rounds (should stop at 2)
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"

        # Final response after round 2 (no tools available)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer after 2 rounds")]
        mock_final_response.stop_reason = "end_turn"

        # Set up API call sequence
        ai_generator_with_mock.client.messages.create.side_effect = [
            mock_tool_response,  # Round 2 wants tools
            mock_final_response,  # Round 2 final response (no tools available)
        ]

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": tool_manager.get_tool_definitions(),
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_tool_response, base_params, tool_manager  # Initial round 1 response
        )

        assert result == "Final answer after 2 rounds"
        # Should make exactly 2 additional API calls
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_sequential_tool_execution_early_termination(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test early termination when Claude doesn't use tools in round 2"""
        # Mock tool use response for round 1
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block]
        mock_round1_response.stop_reason = "tool_use"

        # Mock response for round 2 - Claude chooses not to use tools
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I have sufficient information from the first search")
        ]
        mock_final_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.return_value = mock_final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": tool_manager.get_tool_definitions(),
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_round1_response, base_params, tool_manager
        )

        assert "sufficient information" in result
        # Should make exactly 1 additional API call (round 2 with no tool use)
        assert ai_generator_with_mock.client.messages.create.call_count == 1

    def test_sequential_tool_execution_error_in_round_2(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test error handling when round 2 API call fails"""
        # Mock tool use response for round 1
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        mock_round1_response = Mock()
        mock_round1_response.content = [mock_tool_block]
        mock_round1_response.stop_reason = "tool_use"

        # Mock API error in round 2
        ai_generator_with_mock.client.messages.create.side_effect = Exception(
            "Rate limit exceeded"
        )

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": tool_manager.get_tool_definitions(),
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_round1_response, base_params, tool_manager
        )

        assert (
            "interrupted due to an error" in result or "Rate limit exceeded" in result
        )

    def test_build_round_system_prompt(self, ai_generator_with_mock):
        """Test round-specific system prompt building"""
        base_system = "Base system prompt"

        # Test round 1 prompt
        round1_prompt = ai_generator_with_mock._build_round_system_prompt(
            base_system, 1, 2
        )
        assert "Round 1/2" in round1_prompt
        assert "first opportunity" in round1_prompt

        # Test round 2 prompt
        round2_prompt = ai_generator_with_mock._build_round_system_prompt(
            base_system, 2, 2
        )
        assert "Round 2/2" in round2_prompt
        assert "final tool call opportunity" in round2_prompt

    def test_tool_execution_error_graceful_handling(self, ai_generator_with_mock):
        """Test graceful handling of tool execution errors"""
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "nonexistent_tool"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock tool manager that raises error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool not found")

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Handled error gracefully")]
        mock_final_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
            "tools": [],
        }

        result = ai_generator_with_mock._handle_sequential_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        assert result == "Handled error gracefully"

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
    )
    def test_different_models(self, model_name):
        """Test AIGenerator works with different Claude models"""
        generator = AIGenerator("test-key", model_name)
        assert generator.model == model_name
        assert generator.base_params["model"] == model_name
