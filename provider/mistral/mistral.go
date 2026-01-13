package mistral

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
	defaultBaseURL = "https://api.mistral.ai"
	defaultModel   = "mistral-large-latest"
)

type mistral struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

func New() provider.Provider {
	return &mistral{
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: http.DefaultClient,
	}
}

func (m *mistral) WithAPIKey(key string) provider.Provider {
	m.apiKey = key
	return m
}

func (m *mistral) WithBaseURL(url string) provider.Provider {
	m.baseURL = url
	return m
}

func (m *mistral) WithModel(model string) provider.Provider {
	m.model = model
	return m
}

func (m *mistral) Chat(ctx context.Context, req *provider.ChatRequest) (*provider.ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = m.model
	}

	mistralReq := m.toMistralRequest(req, model)

	body, err := json.Marshal(mistralReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.httpClient.Do(httpReq)
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

	var mistralResp mistralChatCompletionResponse
	if err := json.Unmarshal(respBody, &mistralResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return m.toProviderResponse(&mistralResp), nil
}

func (m *mistral) Stream(ctx context.Context, req *provider.ChatRequest) (*provider.StreamReader, error) {
	model := req.Model
	if model == "" {
		model = m.model
	}

	mistralReq := m.toMistralRequest(req, model)
	mistralReq.Stream = true

	body, err := json.Marshal(mistralReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(httpReq)
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

			var chunk mistralStreamChunk
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

type mistralChatCompletionRequest struct {
	Model            string        `json:"model"`
	Messages         []any         `json:"messages"`
	Temperature      *float64      `json:"temperature,omitempty"`
	TopP             *float64      `json:"top_p,omitempty"`
	MaxTokens        *int          `json:"max_tokens,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
	Stop             []string      `json:"stop,omitempty"`
	RandomSeed       *int          `json:"random_seed,omitempty"`
	Tools            []mistralTool `json:"tools,omitempty"`
	ToolChoice       any           `json:"tool_choice,omitempty"`
	PresencePenalty  *float64      `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64      `json:"frequency_penalty,omitempty"`
}

type mistralMessage struct {
	Role       string            `json:"role"`
	Content    *string           `json:"content,omitempty"`
	ToolCalls  []mistralToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	Name       string            `json:"name,omitempty"`
}

type mistralToolResultMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name,omitempty"`
}

type mistralToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function mistralFunctionCall `json:"function"`
	Index    int                 `json:"index,omitempty"`
}

type mistralFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type mistralTool struct {
	Type     string          `json:"type"`
	Function mistralFunction `json:"function"`
}

type mistralFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

type mistralChatCompletionResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []mistralChoice `json:"choices"`
	Usage   mistralUsage    `json:"usage"`
}

type mistralChoice struct {
	Index        int            `json:"index"`
	Message      mistralMessage `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type mistralUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type mistralStreamChunk struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []mistralStreamChoice `json:"choices"`
}

type mistralStreamChoice struct {
	Index        int                 `json:"index"`
	Delta        mistralDeltaMessage `json:"delta"`
	FinishReason string              `json:"finish_reason"`
}

type mistralDeltaMessage struct {
	Role      string            `json:"role,omitempty"`
	Content   string            `json:"content,omitempty"`
	ToolCalls []mistralToolCall `json:"tool_calls,omitempty"`
}

func (m *mistral) toMistralRequest(req *provider.ChatRequest, model string) *mistralChatCompletionRequest {
	messages := make([]any, len(req.Messages))
	for i, msg := range req.Messages {
		if msg.Role == provider.RoleTool {
			messages[i] = mistralToolResultMessage{
				Role:       string(msg.Role),
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
				Name:       msg.Name,
			}
			continue
		}

		var content *string
		if msg.Content != "" {
			content = &msg.Content
		}

		mistralMsg := mistralMessage{
			Role:       string(msg.Role),
			Content:    content,
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}

		if len(msg.ToolCalls) > 0 {
			mistralMsg.ToolCalls = make([]mistralToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				toolType := tc.Type
				if toolType == "" {
					toolType = "function"
				}
				mistralMsg.ToolCalls[j] = mistralToolCall{
					ID:    tc.ID,
					Type:  toolType,
					Index: tc.Index,
					Function: mistralFunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		messages[i] = mistralMsg
	}

	var tools []mistralTool
	if len(req.Tools) > 0 {
		tools = make([]mistralTool, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = mistralTool{
				Type: t.Type,
				Function: mistralFunction{
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

	return &mistralChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		Stream:           req.Stream,
		Stop:             req.Stop,
		RandomSeed:       req.RandomSeed,
		Tools:            tools,
		ToolChoice:       toolChoice,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
}

func (m *mistral) toProviderResponse(resp *mistralChatCompletionResponse) *provider.ChatResponse {
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
