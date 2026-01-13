package anthropic

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
	defaultBaseURL    = "https://api.anthropic.com"
	defaultModel      = "claude-sonnet-4-20250514"
	apiVersion        = "2023-06-01"
	defaultMaxTokens  = 8192
)

type anthropic struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// New creates a new Anthropic provider.
func New() provider.Provider {
	return &anthropic{
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: http.DefaultClient,
	}
}

func (a *anthropic) WithAPIKey(key string) provider.Provider {
	a.apiKey = key
	return a
}

func (a *anthropic) WithBaseURL(url string) provider.Provider {
	a.baseURL = url
	return a
}

func (a *anthropic) WithModel(model string) provider.Provider {
	a.model = model
	return a
}

func (a *anthropic) Chat(ctx context.Context, req *provider.ChatRequest) (*provider.ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = a.model
	}

	anthropicReq := a.toAnthropicRequest(req, model)

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", apiVersion)

	resp, err := a.httpClient.Do(httpReq)
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

	var anthropicResp anthropicMessageResponse
	if err := json.Unmarshal(respBody, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return a.toProviderResponse(&anthropicResp), nil
}

func (a *anthropic) Stream(ctx context.Context, req *provider.ChatRequest) (*provider.StreamReader, error) {
	model := req.Model
	if model == "" {
		model = a.model
	}

	anthropicReq := a.toAnthropicRequest(req, model)
	anthropicReq.Stream = true

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", apiVersion)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := a.httpClient.Do(httpReq)
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
		var currentToolCallIndex int
		toolCallIndices := make(map[string]int)

		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			var streamEvent anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &streamEvent); err != nil {
				continue
			}

			switch streamEvent.Type {
			case "content_block_delta":
				if streamEvent.Delta != nil {
					switch streamEvent.Delta.Type {
					case "text_delta":
						events <- provider.StreamEvent{
							Delta: provider.Delta{
								Content: streamEvent.Delta.Text,
							},
						}
					case "input_json_delta":
						// Tool call arguments delta
						if streamEvent.Index != nil {
							events <- provider.StreamEvent{
								Delta: provider.Delta{
									ToolCalls: []provider.ToolCall{{
										Index: *streamEvent.Index,
										Function: provider.FunctionCall{
											Arguments: streamEvent.Delta.PartialJSON,
										},
									}},
								},
							}
						}
					}
				}

			case "content_block_start":
				if streamEvent.ContentBlock != nil {
					if streamEvent.ContentBlock.Type == "tool_use" {
						// Start of a tool call
						idx := currentToolCallIndex
						if streamEvent.Index != nil {
							idx = *streamEvent.Index
						}
						toolCallIndices[streamEvent.ContentBlock.ID] = idx
						currentToolCallIndex++

						events <- provider.StreamEvent{
							Delta: provider.Delta{
								ToolCalls: []provider.ToolCall{{
									ID:    streamEvent.ContentBlock.ID,
									Type:  "function",
									Index: idx,
									Function: provider.FunctionCall{
										Name: streamEvent.ContentBlock.Name,
									},
								}},
							},
						}
					}
				}

			case "message_stop":
				events <- provider.StreamEvent{
					FinishReason: "stop",
				}
				return

			case "message_delta":
				if streamEvent.Delta != nil && streamEvent.Delta.StopReason != "" {
					finishReason := streamEvent.Delta.StopReason
					if finishReason == "tool_use" {
						finishReason = "tool_calls"
					}
					events <- provider.StreamEvent{
						FinishReason: finishReason,
					}
				}
			}
		}
	}()

	return provider.NewStreamReader(events, func() { resp.Body.Close() }), nil
}

// Anthropic-specific types

type anthropicMessageRequest struct {
	Model     string             `json:"model"`
	Messages  []anthropicMessage `json:"messages"`
	System    string             `json:"system,omitempty"`
	MaxTokens int                `json:"max_tokens"`
	Stream    bool               `json:"stream,omitempty"`
	Tools     []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string               `json:"role"`
	Content []anthropicContent   `json:"content,omitempty"`
}

type anthropicContent struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicMessageResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Content      []anthropicContent `json:"content"`
	Model        string             `json:"model"`
	StopReason   string             `json:"stop_reason"`
	StopSequence string             `json:"stop_sequence,omitempty"`
	Usage        anthropicUsage     `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicStreamEvent struct {
	Type         string                    `json:"type"`
	Index        *int                      `json:"index,omitempty"`
	Delta        *anthropicDelta           `json:"delta,omitempty"`
	ContentBlock *anthropicContentBlock    `json:"content_block,omitempty"`
	Message      *anthropicMessageResponse `json:"message,omitempty"`
}

type anthropicDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

type anthropicContentBlock struct {
	Type  string `json:"type"`
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
	Text  string `json:"text,omitempty"`
}

func (a *anthropic) toAnthropicRequest(req *provider.ChatRequest, model string) *anthropicMessageRequest {
	var systemPrompt string
	var messages []anthropicMessage

	for _, msg := range req.Messages {
		switch msg.Role {
		case provider.RoleSystem:
			systemPrompt = msg.Content

		case provider.RoleUser:
			messages = append(messages, anthropicMessage{
				Role: "user",
				Content: []anthropicContent{{
					Type: "text",
					Text: msg.Content,
				}},
			})

		case provider.RoleAssistant:
			var content []anthropicContent
			if msg.Content != "" {
				content = append(content, anthropicContent{
					Type: "text",
					Text: msg.Content,
				})
			}
			for _, tc := range msg.ToolCalls {
				var input any
				if tc.Function.Arguments != "" {
					json.Unmarshal([]byte(tc.Function.Arguments), &input)
				}
				content = append(content, anthropicContent{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			if len(content) > 0 {
				messages = append(messages, anthropicMessage{
					Role:    "assistant",
					Content: content,
				})
			}

		case provider.RoleTool:
			messages = append(messages, anthropicMessage{
				Role: "user",
				Content: []anthropicContent{{
					Type:      "tool_result",
					ToolUseID: msg.ToolCallID,
					Content:   msg.Content,
				}},
			})
		}
	}

	var tools []anthropicTool
	for _, t := range req.Tools {
		tools = append(tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	maxTokens := defaultMaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	return &anthropicMessageRequest{
		Model:     model,
		Messages:  messages,
		System:    systemPrompt,
		MaxTokens: maxTokens,
		Tools:     tools,
	}
}

func (a *anthropic) toProviderResponse(resp *anthropicMessageResponse) *provider.ChatResponse {
	var content string
	var toolCalls []provider.ToolCall

	for i, c := range resp.Content {
		switch c.Type {
		case "text":
			content += c.Text
		case "tool_use":
			inputJSON, _ := json.Marshal(c.Input)
			toolCalls = append(toolCalls, provider.ToolCall{
				ID:    c.ID,
				Type:  "function",
				Index: i,
				Function: provider.FunctionCall{
					Name:      c.Name,
					Arguments: string(inputJSON),
				},
			})
		}
	}

	finishReason := resp.StopReason
	if finishReason == "tool_use" {
		finishReason = "tool_calls"
	} else if finishReason == "end_turn" {
		finishReason = "stop"
	}

	return &provider.ChatResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Model:   resp.Model,
		Choices: []provider.Choice{{
			Index: 0,
			Message: provider.Message{
				Role:      provider.RoleAssistant,
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: provider.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
}
