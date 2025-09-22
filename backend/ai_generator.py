from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Multi-Step Tool Usage Guidelines:
- **You can make UP TO 2 tool calls total** across multiple API rounds to answer complex queries
- **Course content questions**: Use search_course_content for specific course materials and content
- **Course outline questions**: Use get_course_outline to retrieve complete course structure, lesson lists, and links
- Use tools strategically - each call should build upon previous results
- If you need more specific information after an initial search, make a second, more targeted tool call

Tool Selection Strategy:
- **First call**: Use broad searches or outline requests to understand scope and gather initial information
- **Second call**: Use specific, targeted searches based on first call results if more detail is needed
- Use get_course_outline for: course structure, lesson lists, course outlines, what lessons are available, course overview
- Use search_course_content for: specific lesson content, detailed materials, answers to content-based questions

Multi-Step Reasoning Examples:
- Complex queries: "Find a course that discusses the same topic as lesson 4 of course X"
  1. First: Get outline of course X to identify lesson 4 topic
  2. Second: Search for courses discussing that specific topic
- Comparison queries: "Compare approaches to vector search between courses"
  1. First: Search for "vector search" across all courses
  2. Second: Get specific course outlines for detailed comparison if needed

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Simple queries**: One tool call may be sufficient
- **Complex queries**: Use up to 2 tool calls to gather comprehensive information
- Analyze results from previous tool calls before deciding on next steps
- Reference previous search results in your reasoning when making additional tool calls
- **No meta-commentary**: Provide direct answers only — no reasoning process, tool explanations, or question-type analysis

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Comprehensive** - Use multiple tool calls when needed for complete answers
5. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude with enhanced error handling
        try:
            response = self.client.messages.create(**api_params)

            # Handle tool execution if needed
            if response.stop_reason == "tool_use" and tool_manager:
                return self._handle_sequential_tool_execution(
                    response, api_params, tool_manager
                )

            # Return direct response
            return response.content[0].text

        except anthropic.BadRequestError as e:
            if "credit balance is too low" in str(e):
                return "I'm sorry, but the chatbot is currently unavailable due to API credit exhaustion. Please contact the administrator to resolve this issue."
            else:
                return f"I encountered an API configuration error. Please contact support with this message: {str(e)}"
        except anthropic.RateLimitError as e:
            return "I'm receiving too many requests right now. Please wait a moment and try again."
        except anthropic.APIError as e:
            return f"I'm experiencing technical difficulties with the AI service. Please try again later. Error: {str(e)}"
        except Exception as e:
            return (
                f"An unexpected error occurred while processing your request: {str(e)}"
            )

    def _handle_sequential_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle sequential tool execution with up to 2 rounds of tool calling.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after all tool execution rounds
        """
        # Initialize conversation tracking
        messages = base_params["messages"].copy()
        current_response = initial_response
        current_round = 1
        max_rounds = 2

        # Process sequential tool call rounds
        while (
            current_round <= max_rounds and current_response.stop_reason == "tool_use"
        ):
            # Add Claude's response (with tool calls) to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this round
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result,
                            }
                        )
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution error: {str(e)}",
                            }
                        )

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare next API call parameters
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": self._build_round_system_prompt(
                    base_params["system"], current_round, max_rounds
                ),
            }

            # Determine if tools should be available for next round
            if current_round < max_rounds:
                # Tools available for potential next call
                next_params["tools"] = base_params.get("tools", [])
                next_params["tool_choice"] = {"type": "auto"}
            else:
                # Final round - no tools, force text response
                pass

            # Make next API call
            try:
                current_response = self.client.messages.create(**next_params)
                current_round += 1

                # If Claude doesn't use tools, we're done
                if current_response.stop_reason != "tool_use":
                    break

            except Exception as e:
                # Error handling - return best available response
                if current_round == 1:
                    return f"Error during tool execution: {str(e)}"
                else:
                    # Try to salvage a response from partial conversation
                    return self._salvage_partial_response(messages, str(e))

        # Return final response
        return current_response.content[0].text

    def _build_round_system_prompt(
        self, base_system: str, current_round: int, max_rounds: int
    ) -> str:
        """Build system prompt with round-specific guidance"""

        # Extract the base prompt without round-specific additions
        base_prompt = base_system

        # Add round-specific guidance
        if current_round == 1:
            round_guidance = f"\n\nRound {current_round}/{max_rounds}: This is your first opportunity to use tools. Consider if you need broad or specific information."
        elif current_round == 2:
            round_guidance = f"\n\nRound {current_round}/{max_rounds}: This is your final tool call opportunity. Make it count - be specific and targeted based on previous results."
        else:
            round_guidance = f"\n\nRound {current_round}/{max_rounds}: No more tool calls available. Provide your final answer based on all previous information."

        return base_prompt + round_guidance

    def _salvage_partial_response(self, messages: list, error_msg: str) -> str:
        """Extract best available response from partial conversation"""

        # Look for any text responses in the conversation
        for message in reversed(messages):
            if message.get("role") == "assistant":
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if (
                            hasattr(block, "type")
                            and block.type == "text"
                            and len(block.text.strip()) > 20
                        ):
                            return f"{block.text}\n\n(Note: Additional information gathering was interrupted due to an error: {error_msg})"
                elif isinstance(content, str) and len(content.strip()) > 20:
                    return f"{content}\n\n(Note: Additional information gathering was interrupted due to an error: {error_msg})"

        return f"I encountered an error while processing your request and was unable to provide a complete response: {error_msg}"
