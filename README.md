# llm-translate

Zero-dependency Go library for bidirectional protocol translation between the Anthropic Messages API, OpenAI Chat Completions, and the OpenAI Responses API.

## Features

- **Anthropic ↔ OpenAI Request & Response:** Full fidelity translation of messages, tool definitions, tool results, thinking/reasoning blocks, system prompts, and multi-part vision content.
- **SSE Stream Translation:** Real-time translation between OpenAI chunk deltas and Anthropic SSE events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`).
- **OpenAI Responses ↔ Chat:** Bidirectional translation for newer OpenAI Responses API backends to standard Chat completions.
- **Pure Standard Library:** Zero CGO, zero external dependencies.

## Installation

```bash
go get github.com/xibodev/llm-translate
```

## Quick Start

### Convert Anthropic Messages Request to OpenAI

```go
import "github.com/xibodev/llm-translate"

openaiMessages := translate.AnthropicMessagesToOpenAI(req.Messages, req.System)
tools := translate.AnthropicToolsToOpenAI(req.Tools)
```

### Stream OpenAI Chunks as Anthropic SSE

```go
translate.OpenAIStreamToAnthropicSSE(chunksIter, "claude-3-5-sonnet", func(sseEvent string) {
    w.Write([]byte(sseEvent))
    w.(http.Flusher).Flush()
})
```

## License

MIT
