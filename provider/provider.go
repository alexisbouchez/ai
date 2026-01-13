package provider

import (
	"context"
	"errors"
)

type Provider interface {
	WithAPIKey(key string) Provider
	WithBaseURL(url string) Provider
	WithModel(model string) Provider
	Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	Stream(ctx context.Context, req *ChatRequest) (*StreamReader, error)
}

type StreamReader struct {
	events chan StreamEvent
	err    error
	done   bool
	close  func()
}

func NewStreamReader(events chan StreamEvent, close func()) *StreamReader {
	return &StreamReader{events: events, close: close}
}

func (s *StreamReader) Recv() (StreamEvent, error) {
	if s.done {
		return StreamEvent{}, ErrStreamClosed
	}
	event, ok := <-s.events
	if !ok {
		s.done = true
		return StreamEvent{}, ErrStreamClosed
	}
	if event.Err != nil {
		s.err = event.Err
		return event, event.Err
	}
	return event, nil
}

func (s *StreamReader) Close() {
	if s.close != nil {
		s.close()
	}
}

type StreamEvent struct {
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
	Err          error  `json:"-"`
}

type Delta struct {
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

var ErrStreamClosed = errors.New("stream closed")

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
	Index    int          `json:"index,omitempty"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceAny      ToolChoice = "any"
	ToolChoiceRequired ToolChoice = "required"
)

type ChatRequest struct {
	Messages         []Message   `json:"messages"`
	Model            string      `json:"model,omitempty"`
	Temperature      *float64    `json:"temperature,omitempty"`
	TopP             *float64    `json:"top_p,omitempty"`
	MaxTokens        *int        `json:"max_tokens,omitempty"`
	Stream           bool        `json:"stream,omitempty"`
	Stop             []string    `json:"stop,omitempty"`
	Tools            []Tool      `json:"tools,omitempty"`
	ToolChoice       *ToolChoice `json:"tool_choice,omitempty"`
	PresencePenalty  *float64    `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64    `json:"frequency_penalty,omitempty"`
	RandomSeed       *int        `json:"random_seed,omitempty"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

const (
	FinishReasonStop        = "stop"
	FinishReasonLength      = "length"
	FinishReasonToolCalls   = "tool_calls"
	FinishReasonModelLength = "model_length"
	FinishReasonError       = "error"
)

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
