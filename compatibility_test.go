package translate

import (
	"errors"
	"reflect"
	"testing"
)

func TestResponsesRequestReportUnsupportedMaterialField(t *testing.T) {
	result, err := ResponsesRequestToChatWithReport(map[string]any{
		"input": "hello",
		"text":  map[string]any{"format": map[string]any{"type": "json_schema"}},
	})
	if err == nil {
		t.Fatal("expected existing converter to reject unsupported field")
	}
	want := []Loss{{Path: "text", Class: LossUnsupported, Severity: LossMaterial, Detail: "Responses field requires a native Responses model"}}
	if !reflect.DeepEqual(result.Report.Losses, want) {
		t.Fatalf("losses = %#v, want %#v", result.Report.Losses, want)
	}
	var materialError *MaterialLossError
	if !errors.As(result.RejectMaterialLoss(), &materialError) {
		t.Fatal("material report was not rejected with MaterialLossError")
	}
}

func TestChatToResponsesReportAdvisoryRenameAndApproximation(t *testing.T) {
	result := ChatToResponsesWithReport("model", []map[string]any{{
		"role": "assistant",
		"tool_calls": []any{map[string]any{
			"id": "call_1", "function": map[string]any{"name": "lookup", "arguments": "{}"},
		}},
	}}, map[string]any{"max_tokens": 10}, false)
	want := []Loss{
		{Path: "max_tokens", Class: LossRenamed, Severity: LossAdvisory, Detail: "converted to max_output_tokens"},
		{Path: "messages.0.tool_calls.0", Class: LossApproximated, Severity: LossAdvisory, Detail: "missing tool output is synthesized as an empty string"},
	}
	if !reflect.DeepEqual(result.Report.Losses, want) {
		t.Fatalf("losses = %#v, want %#v", result.Report.Losses, want)
	}
	if err := result.RejectMaterialLoss(); err != nil {
		t.Fatalf("advisory losses rejected: %v", err)
	}
}

func TestAnthropicRequestReportNoLoss(t *testing.T) {
	result := AnthropicRequestToOpenAIWithReport(map[string]any{
		"model": "model", "messages": []any{map[string]any{"role": "user", "content": "hello"}},
	})
	if len(result.Report.Losses) != 0 {
		t.Fatalf("unexpected losses: %#v", result.Report.Losses)
	}
	if len(result.Value.Messages) != 1 || result.Value.Messages[0]["content"] != "hello" {
		t.Fatalf("unexpected conversion: %#v", result.Value)
	}
}

func TestReportOrderingIsDeterministic(t *testing.T) {
	report := NewReport(
		material("z", LossDropped, "last"),
		advisory("a", LossRenamed, "second"),
		material("a", LossUnsupported, "first material"),
		advisory("a", LossRenamed, "second"),
	)
	want := []Loss{
		{Path: "a", Class: LossRenamed, Severity: LossAdvisory, Detail: "second"},
		{Path: "a", Class: LossUnsupported, Severity: LossMaterial, Detail: "first material"},
		{Path: "z", Class: LossDropped, Severity: LossMaterial, Detail: "last"},
	}
	if !reflect.DeepEqual(report.Losses, want) {
		t.Fatalf("losses = %#v, want %#v", report.Losses, want)
	}
}
