# GorAI LLM Client

A unified LLM client library supporting multiple providers (OpenAI, Anthropic, etc.) with advanced features like tool calling, streaming responses, and automatic conversation loops.

## Version

**Current Version: 0.3.2**

## Features

- 🔌 **Multiple LLM Providers**: Support for OpenAI, Anthropic, DeepSeek, and Minimax APIs
- 🔧 **Tool Calling**: Built-in support for function/tool calling with automatic execution
- 🌊 **Streaming Responses**: Real-time streaming for better user experience
- 🔄 **Conversation Loops**: Automatic multi-turn conversations with `chatToNextLoop`
- 💭 **Thinking Support**: Handle model reasoning/thinking content (Anthropic extended thinking)
- 🧠 **Interleaved Thinking Router**: Support for DeepSeek OpenAI-compatible and Minimax Anthropic-compatible interleaved thinking interactions
- 🖼️ **Image Support**: Built-in support for image format handling
- 🎯 **Unified Interface**: Consistent API across different providers
- 🛠️ **Flexible Tool Execution**: Custom tool executors with `ToolExecutor` interface

## Examples

Check the `examples/` directory for complete examples:
- `chat_with_tools_example.py`: Comprehensive examples of tool calling

## Project Structure

```
GorAI_LLMClient/
├── models/
│   ├── __init__.py                    # create_model factory function
│   ├── _model_base.py                 # Base model class
│   ├── _openai_model.py               # OpenAI implementation
│   ├── _anthropic_model.py            # Anthropic implementation
│   ├── _deepseek_openai_model.py      # DeepSeek OpenAI-compatible implementation
│   └── _minimax_anthropic_model.py    # Minimax Anthropic-compatible implementation
├── message/
│   ├── __init__.py
│   └── _message_base.py      # MsgReturn message format
├── executor.py               # Tool executor interfaces
└── examples/
    └── chat_with_tools_example.py
```

## License

MIT License