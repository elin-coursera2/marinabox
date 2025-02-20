#!/usr/bin/env python3
import asyncio
import argparse
from anthropic import Anthropic
from .tools import ToolCollection, ComputerTool, BashTool, EditTool
from .loop import sampling_loop

async def main(prompt: str, aws_access_key_id: str, aws_secret_access_key: str, aws_session_token: str, aws_region: str, port: int = 8002):
    responses = []  # Create a list to store responses
    
    def output_callback(content):
        if content["type"] == "text":
            responses.append(("text", content['text']))
            print(f"Assistant: {content['text']}")
        elif content["type"] == "tool_use":
            responses.append(("tool_use", content['name'], content['input']))
            print(f"Tool use: {content['name']} with input {content['input']}")

    def tool_output_callback(result, tool_id):
        if result.output:
            responses.append(("tool_output", result.output))
            print(f"Tool output: {result.output}")
        if result.base64_image:
            responses.append(("tool_output_image", result.base64_image))
            print(f"Tool output: IMAGE")
        if result.error:
            responses.append(("tool_error", result.error))
            print(f"Tool error: {result.error}")

    def api_response_callback(request, response, error):
        if error:
            print(f"API error: {error}")

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    computer_tool = ComputerTool(port=port)
    bash_tool = BashTool()
    edit_tool = EditTool()
    
    tools = ToolCollection(computer_tool, bash_tool, edit_tool)

    messages = await sampling_loop(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="bedrock",
        system_prompt_suffix="",
        messages=messages,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        api_response_callback=api_response_callback,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
        tools=tools,
        max_iterations=20
    )
    
    return responses  # Return the collected responses

if __name__ == "__main__":
    asyncio.run(main())