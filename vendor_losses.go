package translate

import (
	"fmt"
	"strconv"
)

// Vendor fields travel inside messages, where the conversions historically
// dropped them without a trace. These helpers report each one at its exact
// source path so a LossPolicy can target it:
//
//   - A dropped Gemini thought signature is material: without it the next
//     turn of a multi-turn tool call fails or loses the model's reasoning.
//   - Dropped reasoning_details and cache_control are advisory: the answer
//     is intact, and a product that depends on them can still reject the
//     loss by path.
//   - Dropped reasoning items and blocks are reported at <item path>.reasoning
//     so that a policy can match every reasoning loss with **.reasoning.

const thoughtSignaturePath = "extra_content.google.thought_signature"

// present reports whether a vendor value carries anything. Null, "" and
// empty arrays or objects carry nothing, so dropping them loses nothing.
func present(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	}
	return true
}

func hasThoughtSignature(call map[string]any) bool {
	extra, _ := asMap(call["extra_content"])
	google, _ := asMap(extra["google"])
	return present(google["thought_signature"])
}

func droppedThoughtSignature(callPath, target string) Loss {
	return material(callPath+"."+thoughtSignaturePath, LossDropped, "Gemini thought signature is not representable by "+target+", so the next turn cannot replay it")
}

func droppedReasoningDetails(path, target string) Loss {
	return advisory(path, LossDropped, "reasoning_details are not representable by "+target)
}

func droppedCacheControl(path, target string) Loss {
	return advisory(path, LossDropped, "cache_control is not representable by "+target)
}

func droppedReasoning(path string) Loss {
	return advisory(path+".reasoning", LossDropped, "reasoning is not emitted by Chat")
}

func isThinkingBlock(block map[string]any) bool {
	return block["type"] == "thinking" || block["type"] == "redacted_thinking"
}

// chatMessageVendorLosses reports the thought signatures and
// reasoning_details that a conversion of one Chat message to target drops.
// prefix is the message path, such as messages.3 or choices.0.message.
func chatMessageVendorLosses(prefix string, message map[string]any, target string) []Loss {
	var losses []Loss
	if calls, ok := asList(message["tool_calls"]); ok {
		for j, raw := range calls {
			if call, ok := asMap(raw); ok && hasThoughtSignature(call) {
				losses = append(losses, droppedThoughtSignature(prefix+".tool_calls."+strconv.Itoa(j), target))
			}
		}
	}
	if present(message["reasoning_details"]) {
		losses = append(losses, droppedReasoningDetails(prefix+".reasoning_details", target))
	}
	return losses
}

// contentPartCacheControls returns the paths of the content parts of one
// Chat message that carry cache_control, split by whether the part is a text
// part.
func contentPartCacheControls(prefix string, content any) (text, other []string) {
	parts, _ := asList(content)
	for k, raw := range parts {
		part, ok := asMap(raw)
		if !ok || !present(part["cache_control"]) {
			continue
		}
		path := fmt.Sprintf("%s.content.%d.cache_control", prefix, k)
		if _, isText := part["text"].(string); isText && part["type"] == "text" {
			text = append(text, path)
		} else {
			other = append(other, path)
		}
	}
	return text, other
}
