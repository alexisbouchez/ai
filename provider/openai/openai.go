package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/alexisbouchez/ai/provider"
)

const (
	defaultBaseURL = "https://api.openai.com"
	defaultModel   = "gpt-4o"
)

type openai struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// New creates a new OpenAI provider.
func New() provider.Provider {
	return &openai{
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: http.DefaultClient,
	}
}

func (o *openai) WithAPIKey(key string) provider.Provider {
	o.apiKey = key
	return o
}

func (o *openai) WithBaseURL(url string) provider.Provider {
	o.baseURL = url
	return o
}

func (o *openai) WithModel(model string) provider.Provider {
	o.model = model
	return o
}

func (o *openai) Chat(ctx context.Context, req *provider.ChatRequest) (*provider.ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = o.model
	}

	openaiReq := o.toOpenAIRequest(req, model)

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var openaiResp openaiChatCompletionResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return o.toProviderResponse(&openaiResp), nil
}

func (o *openai) Stream(ctx context.Context, req *provider.ChatRequest) (*provider.StreamReader, error) {
	model := req.Model
	if model == "" {
		model = o.model
	}

	openaiReq := o.toOpenAIRequest(req, model)
	openaiReq.Stream = true

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := o.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	events := make(chan provider.StreamEvent)

	go func() {
		defer close(events)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return
			}

			var chunk openaiStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				events <- provider.StreamEvent{Err: fmt.Errorf("failed to parse chunk: %w", err)}
				return
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			event := provider.StreamEvent{
				Delta: provider.Delta{
					Content: choice.Delta.Content,
				},
				FinishReason: choice.FinishReason,
			}

			if len(choice.Delta.ToolCalls) > 0 {
				event.Delta.ToolCalls = make([]provider.ToolCall, len(choice.Delta.ToolCalls))
				for i, tc := range choice.Delta.ToolCalls {
					event.Delta.ToolCalls[i] = provider.ToolCall{
						ID:    tc.ID,
						Type:  tc.Type,
						Index: tc.Index,
						Function: provider.FunctionCall{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					}
				}
			}

			select {
			case events <- event:
			case <-ctx.Done():
				return
			}
		}
	}()

	return provider.NewStreamReader(events, func() { resp.Body.Close() }), nil
}

// OpenAI-specific request/response types

type openaiChatCompletionRequest struct {
	Model            string       `json:"model"`
	Messages         []any        `json:"messages"`
	Temperature      *float64     `json:"temperature,omitempty"`
	TopP             *float64     `json:"top_p,omitempty"`
	MaxTokens        *int         `json:"max_tokens,omitempty"`
	Stream           bool         `json:"stream,omitempty"`
	Stop             []string     `json:"stop,omitempty"`
	Tools            []openaiTool `json:"tools,omitempty"`
	ToolChoice       any          `json:"tool_choice,omitempty"`
	PresencePenalty  *float64     `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64     `json:"frequency_penalty,omitempty"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    *string          `json:"content,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type openaiToolResultMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiFunctionCall `json:"function"`
	Index    int                `json:"index,omitempty"`
}

type openaiFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string         `json:"type"`
	Function openaiFunction `json:"function"`
}

type openaiFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

type openaiChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Index        int           `json:"index"`
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openaiStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openaiStreamChoice `json:"choices"`
}

type openaiStreamChoice struct {
	Index        int                `json:"index"`
	Delta        openaiDeltaMessage `json:"delta"`
	FinishReason string             `json:"finish_reason"`
}

type openaiDeltaMessage struct {
	Role      string           `json:"role,omitempty"`
	Content   string           `json:"content,omitempty"`
	ToolCalls []openaiToolCall `json:"tool_calls,omitempty"`
}

func (o *openai) toOpenAIRequest(req *provider.ChatRequest, model string) *openaiChatCompletionRequest {
	messages := make([]any, len(req.Messages))
	for i, msg := range req.Messages {
		if msg.Role == provider.RoleTool {
			messages[i] = openaiToolResultMessage{
				Role:       string(msg.Role),
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
			}
			continue
		}

		var content *string
		if msg.Content != "" {
			content = &msg.Content
		}

		openaiMsg := openaiMessage{
			Role:       string(msg.Role),
			Content:    content,
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}

		if len(msg.ToolCalls) > 0 {
			openaiMsg.ToolCalls = make([]openaiToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				toolType := tc.Type
				if toolType == "" {
					toolType = "function"
				}
				openaiMsg.ToolCalls[j] = openaiToolCall{
					ID:    tc.ID,
					Type:  toolType,
					Index: tc.Index,
					Function: openaiFunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		messages[i] = openaiMsg
	}

	var tools []openaiTool
	if len(req.Tools) > 0 {
		tools = make([]openaiTool, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = openaiTool{
				Type: t.Type,
				Function: openaiFunction{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
					Strict:      t.Function.Strict,
				},
			}
		}
	}

	var toolChoice any
	if req.ToolChoice != nil {
		toolChoice = string(*req.ToolChoice)
	}

	return &openaiChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		Stream:           req.Stream,
		Stop:             req.Stop,
		Tools:            tools,
		ToolChoice:       toolChoice,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
}

func (o *openai) toProviderResponse(resp *openaiChatCompletionResponse) *provider.ChatResponse {
	choices := make([]provider.Choice, len(resp.Choices))
	for i, c := range resp.Choices {
		var toolCalls []provider.ToolCall
		if len(c.Message.ToolCalls) > 0 {
			toolCalls = make([]provider.ToolCall, len(c.Message.ToolCalls))
			for j, tc := range c.Message.ToolCalls {
				toolType := tc.Type
				if toolType == "" {
					toolType = "function"
				}
				toolCalls[j] = provider.ToolCall{
					ID:    tc.ID,
					Type:  toolType,
					Index: tc.Index,
					Function: provider.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		var content string
		if c.Message.Content != nil {
			content = *c.Message.Content
		}

		choices[i] = provider.Choice{
			Index: c.Index,
			Message: provider.Message{
				Role:       provider.Role(c.Message.Role),
				Content:    content,
				ToolCalls:  toolCalls,
				ToolCallID: c.Message.ToolCallID,
				Name:       c.Message.Name,
			},
			FinishReason: c.FinishReason,
		}
	}

	return &provider.ChatResponse{
		ID:      resp.ID,
		Object:  resp.Object,
		Created: resp.Created,
		Model:   resp.Model,
		Choices: choices,
		Usage: provider.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}
