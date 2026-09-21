package translate

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// OpenAIChatRequest is the two-part request value returned by Chat conversions.
type OpenAIChatRequest struct {
	Messages []map[string]any
	Keywords map[string]any
}

// AnthropicMessages is the two-part request value returned by Anthropic conversions.
type AnthropicMessages struct {
	System   string
	Messages []map[string]any
}

// AnthropicRequestToOpenAIWithReport delegates to AnthropicRequestToOpenAI and
// gives its incompatible paths structured material-loss semantics.
func AnthropicRequestToOpenAIWithReport(payload map[string]any) ConversionResult[OpenAIChatRequest] {
	messages, keywords, incompatible := AnthropicRequestToOpenAI(payload)
	losses := make([]Loss, 0, len(incompatible))
	for _, path := range incompatible {
		losses = append(losses, material(path, LossUnsupported, "value is not representable by OpenAI Chat"))
	}
	return ConversionResult[OpenAIChatRequest]{
		Value:  OpenAIChatRequest{Messages: messages, Keywords: keywords},
		Report: NewReport(losses...),
	}
}

// OpenAIMessagesToAnthropicWithReport converts messages and reports known
// normalizations performed by the existing converter.
func OpenAIMessagesToAnthropicWithReport(messages []map[string]any) ConversionResult[AnthropicMessages] {
	system, converted := OpenAIMessagesToAnthropic(messages)
	var losses []Loss
	for i, message := range messages {
		path := "messages." + strconv.Itoa(i)
		role, _ := message["role"].(string)
		if role == "developer" {
			losses = append(losses, advisory(path+".role", LossApproximated, "developer instruction is merged into Anthropic system text"))
		}
		if role != "system" && role != "developer" && role != "assistant" && role != "tool" && role != "user" && role != "" {
			losses = append(losses, material(path+".role", LossApproximated, "role is converted to user"))
		}
		if role == "assistant" {
			if calls, ok := asList(message["tool_calls"]); ok {
				for j, raw := range calls {
					call, _ := asMap(raw)
					fn, _ := asMap(call["function"])
					args, _ := fn["arguments"].(string)
					var object map[string]any
					if args != "" && json.Unmarshal([]byte(args), &object) != nil {
						losses = append(losses, material(fmt.Sprintf("%s.tool_calls.%d.function.arguments", path, j), LossApproximated, "invalid or non-object JSON arguments become an empty object"))
					}
				}
			}
		}
	}
	return ConversionResult[AnthropicMessages]{Value: AnthropicMessages{System: system, Messages: converted}, Report: NewReport(losses...)}
}

// OpenAIResponseToAnthropicWithReport converts a Chat response with fidelity findings.
func OpenAIResponseToAnthropicWithReport(response map[string]any, model string) ConversionResult[map[string]any] {
	var losses []Loss
	if choices, ok := asList(response["choices"]); ok && len(choices) > 1 {
		losses = append(losses, material("choices.1", LossDropped, "only the first Chat choice is converted"))
	}
	if choices, ok := asList(response["choices"]); ok && len(choices) > 0 {
		choice, _ := asMap(choices[0])
		if choice["finish_reason"] == "content_filter" {
			losses = append(losses, advisory("choices.0.finish_reason", LossApproximated, "content_filter maps to Anthropic end_turn"))
		}
	}
	return ConversionResult[map[string]any]{Value: OpenAIResponseToAnthropic(response, model), Report: NewReport(losses...)}
}

// AnthropicResponseToOpenAIWithReport converts an Anthropic response with fidelity findings.
func AnthropicResponseToOpenAIWithReport(response map[string]any, model string) ConversionResult[map[string]any] {
	var losses []Loss
	if blocks, ok := asList(response["content"]); ok {
		for i, raw := range blocks {
			block, _ := asMap(raw)
			if kind, _ := block["type"].(string); kind != "text" && kind != "tool_use" {
				losses = append(losses, material(fmt.Sprintf("content.%d", i), LossDropped, "Anthropic content block is not representable by Chat"))
			}
		}
	}
	return ConversionResult[map[string]any]{Value: AnthropicResponseToOpenAI(response, model), Report: NewReport(losses...)}
}

// ChatToResponsesWithReport converts a Chat request and reports known fallback synthesis.
func ChatToResponsesWithReport(model string, messages []map[string]any, kw map[string]any, stream bool) ConversionResult[map[string]any] {
	var losses []Loss
	if kw["max_completion_tokens"] != nil {
		losses = append(losses, advisory("max_completion_tokens", LossRenamed, "converted to max_output_tokens"))
	} else if kw["max_tokens"] != nil {
		losses = append(losses, advisory("max_tokens", LossRenamed, "converted to max_output_tokens"))
	}
	supported := map[string]bool{"max_completion_tokens": true, "max_tokens": true, "reasoning_effort": true, "temperature": true, "top_p": true, "tools": true, "tool_choice": true, "metadata": true}
	for key, value := range kw {
		if value != nil && !supported[key] {
			losses = append(losses, material(key, LossDropped, "Chat option is not emitted in the Responses request"))
		}
	}
	outputs := map[string]bool{}
	for _, message := range messages {
		if message["role"] == "tool" {
			if id, _ := message["tool_call_id"].(string); id != "" {
				outputs[id] = true
			}
		}
	}
	for i, message := range messages {
		if calls, ok := asList(message["tool_calls"]); ok {
			for j, raw := range calls {
				call, _ := asMap(raw)
				id, _ := call["id"].(string)
				if id != "" && !outputs[id] {
					losses = append(losses, advisory(fmt.Sprintf("messages.%d.tool_calls.%d", i, j), LossApproximated, "missing tool output is synthesized as an empty string"))
				}
			}
		}
	}
	return ConversionResult[map[string]any]{Value: ChatToResponses(model, messages, kw, stream), Report: NewReport(losses...)}
}

// ResponsesRequestToChatWithReport preserves existing validation errors and
// reports unsupported top-level fields as material compatibility loss.
func ResponsesRequestToChatWithReport(payload map[string]any) (ConversionResult[OpenAIChatRequest], error) {
	known := map[string]bool{"model": true, "input": true, "instructions": true, "stream": true, "max_output_tokens": true, "reasoning": true, "temperature": true, "top_p": true, "tools": true, "tool_choice": true, "metadata": true, "store": true, "background": true, "force_api_support": true, "previous_response_id": true, "conversation": true, "client_metadata": true, "include": true, "parallel_tool_calls": true, "prompt_cache_key": true}
	var losses []Loss
	for key, value := range payload {
		if value != nil && !known[key] {
			losses = append(losses, material(key, LossUnsupported, "Responses field requires a native Responses model"))
		}
	}
	if payload["previous_response_id"] != nil && payload["previous_response_id"] != "" {
		losses = append(losses, material("previous_response_id", LossUnsupported, "server-side response state is not representable by Chat"))
	}
	if payload["conversation"] != nil {
		losses = append(losses, material("conversation", LossUnsupported, "conversation state is not representable by Chat"))
	}
	if values, ok := asList(payload["include"]); ok && len(values) > 0 {
		losses = append(losses, material("include", LossUnsupported, "included Responses-only data is not representable by Chat"))
	}
	messages, keywords, err := ResponsesRequestToChat(payload)
	if err != nil && len(losses) == 0 {
		losses = append(losses, material("$", LossUnsupported, err.Error()))
	}
	return ConversionResult[OpenAIChatRequest]{Value: OpenAIChatRequest{Messages: messages, Keywords: keywords}, Report: NewReport(losses...)}, err
}

// ResponsesToChatWithReport converts a Responses object with fidelity findings.
func ResponsesToChatWithReport(model string, response map[string]any) ConversionResult[map[string]any] {
	var losses []Loss
	if output, ok := asList(response["output"]); ok {
		for i, raw := range output {
			item, _ := asMap(raw)
			kind, _ := item["type"].(string)
			if kind == "reasoning" {
				losses = append(losses, advisory(fmt.Sprintf("output.%d", i), LossDropped, "Responses reasoning content is not emitted by Chat"))
			} else if kind != "message" && kind != "function_call" {
				losses = append(losses, material(fmt.Sprintf("output.%d", i), LossDropped, "Responses output item is not representable by Chat"))
			}
			if kind == "message" {
				if content, ok := asList(item["content"]); ok {
					for j, rawPart := range content {
						part, _ := asMap(rawPart)
						if part["type"] != "output_text" {
							losses = append(losses, material(fmt.Sprintf("output.%d.content.%d", i, j), LossDropped, "Responses content part is not representable by Chat"))
						} else if annotations, ok := asList(part["annotations"]); ok && len(annotations) > 0 {
							losses = append(losses, material(fmt.Sprintf("output.%d.content.%d.annotations", i, j), LossDropped, "text annotations are not emitted by Chat"))
						}
					}
				}
			}
		}
	}
	if response["status"] == "incomplete" && !responsesIncompleteMaxTokens(response) {
		losses = append(losses, material("incomplete_details.reason", LossApproximated, "Chat cannot preserve this incomplete reason"))
	}
	return ConversionResult[map[string]any]{Value: ResponsesToChat(model, response), Report: NewReport(losses...)}
}

// ChatResponseToResponsesWithReport converts a Chat response with fidelity findings.
func ChatResponseToResponsesWithReport(model string, chat map[string]any) ConversionResult[map[string]any] {
	return ChatResponseToResponsesWithRequestAndReport(model, chat, nil)
}

// ChatResponseToResponsesWithRequestAndReport also uses request fields when
// constructing the Responses envelope.
func ChatResponseToResponsesWithRequestAndReport(model string, chat map[string]any, request map[string]any) ConversionResult[map[string]any] {
	var losses []Loss
	if choices, ok := asList(chat["choices"]); ok && len(choices) > 1 {
		losses = append(losses, material("choices.1", LossDropped, "only the first Chat choice is converted"))
	}
	return ConversionResult[map[string]any]{Value: ChatResponseToResponsesWithRequest(model, chat, request), Report: NewReport(losses...)}
}

// ResponsesToChatChunksWithReport reports that a complete response is rendered
// as buffered chunks rather than preserving original event boundaries.
func ResponsesToChatChunksWithReport(model string, response map[string]any) ConversionResult[[]string] {
	base := ResponsesToChatWithReport(model, response)
	losses := append([]Loss(nil), base.Report.Losses...)
	losses = append(losses, advisory("stream", LossApproximated, "buffered response is rendered as synthetic Chat chunks"))
	return ConversionResult[[]string]{Value: ResponsesToChatChunks(model, response), Report: NewReport(losses...)}
}

// OpenAIStreamToAnthropicSSEWithReport delegates stream conversion and reports
// the token estimate used when chunks do not carry usage.
func OpenAIStreamToAnthropicSSEWithReport(chunks func() (string, bool), model string, emit func(string)) Report {
	sawText := false
	sawOutputUsage := false
	observed := func() (string, bool) {
		raw, ok := chunks()
		if !ok {
			return raw, false
		}
		var chunk map[string]any
		if json.Unmarshal([]byte(raw), &chunk) == nil {
			if usage, ok := asMap(chunk["usage"]); ok && usage["completion_tokens"] != nil {
				sawOutputUsage = true
			}
			if choices, ok := asList(chunk["choices"]); ok {
				for _, rawChoice := range choices {
					choice, _ := asMap(rawChoice)
					delta, _ := asMap(choice["delta"])
					if text, _ := delta["content"].(string); text != "" {
						sawText = true
					}
				}
			}
		}
		return raw, true
	}
	OpenAIStreamToAnthropicSSE(observed, model, emit)
	if sawText && !sawOutputUsage {
		return NewReport(advisory("usage.output_tokens", LossApproximated, "output tokens are estimated from streamed text because usage is absent"))
	}
	return NewReport()
}

// AnthropicSSEToOpenAIChunksWithReport delegates stream conversion and reports
// usage fields that the existing Chat chunk adapter does not emit.
func AnthropicSSEToOpenAIChunksWithReport(lines func() (string, bool), model string, emit func(string)) Report {
	sawUsage := false
	observed := func() (string, bool) {
		raw, ok := lines()
		if !ok {
			return raw, false
		}
		line := raw
		if len(line) >= len("data:") && line[:len("data:")] == "data:" {
			var event map[string]any
			if json.Unmarshal([]byte(line[len("data:"):]), &event) == nil && event["usage"] != nil {
				sawUsage = true
			}
		}
		return raw, true
	}
	AnthropicSSEToOpenAIChunks(observed, model, emit)
	if sawUsage {
		return NewReport(material("usage", LossDropped, "Anthropic streaming usage is not emitted in Chat chunks"))
	}
	return NewReport()
}
