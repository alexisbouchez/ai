package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/palmframe/palm/provider"
)

const (
	defaultBaseURL = "http://localhost:11434"
	defaultModel   = "llama3.2"
)

type ollama struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

func New() provider.Provider {
	return &ollama{
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: http.DefaultClient,
	}
}

func (o *ollama) WithAPIKey(key string) provider.Provider {
	return o
}

func (o *ollama) WithBaseURL(url string) provider.Provider {
	o.baseURL = url
	return o
}

func (o *ollama) WithModel(model string) provider.Provider {
	o.model = model
	return o
}

func (o *ollama) getClient() (*api.Client, error) {
	u, err := url.Parse(o.baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}
	return api.NewClient(u, o.httpClient), nil
}

func (o *ollama) Chat(ctx context.Context, req *provider.ChatRequest) (*provider.ChatResponse, error) {
	client, err := o.getClient()
	if err != nil {
		return nil, err
	}

	model := req.Model
	if model == "" {
		model = o.model
	}

	chatReq := &api.ChatRequest{
		Model:    model,
		Messages: o.convertMessages(req.Messages),
		Stream:   boolPtr(false),
	}

	if len(req.Tools) > 0 {
		chatReq.Tools = o.convertTools(req.Tools)
	}

	opts := map[string]any{}
	if req.Temperature != nil {
		opts["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		opts["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		opts["num_predict"] = *req.MaxTokens
	}
	if len(req.Stop) > 0 {
		opts["stop"] = req.Stop
	}
	if req.RandomSeed != nil {
		opts["seed"] = *req.RandomSeed
	}
	if len(opts) > 0 {
		chatReq.Options = opts
	}

	var response *api.ChatResponse
	err = client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
		response = &resp
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("chat request failed: %w", err)
	}

	return o.toProviderResponse(response, model), nil
}

func (o *ollama) Stream(ctx context.Context, req *provider.ChatRequest) (*provider.StreamReader, error) {
	client, err := o.getClient()
	if err != nil {
		return nil, err
	}

	model := req.Model
	if model == "" {
		model = o.model
	}

	chatReq := &api.ChatRequest{
		Model:    model,
		Messages: o.convertMessages(req.Messages),
		Stream:   boolPtr(true),
	}

	if len(req.Tools) > 0 {
		chatReq.Tools = o.convertTools(req.Tools)
	}

	opts := map[string]any{}
	if req.Temperature != nil {
		opts["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		opts["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		opts["num_predict"] = *req.MaxTokens
	}
	if len(req.Stop) > 0 {
		opts["stop"] = req.Stop
	}
	if req.RandomSeed != nil {
		opts["seed"] = *req.RandomSeed
	}
	if len(opts) > 0 {
		chatReq.Options = opts
	}

	events := make(chan provider.StreamEvent)
	done := make(chan struct{})

	go func() {
		defer close(events)

		err := client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
			var toolCalls []provider.ToolCall
			for i, tc := range resp.Message.ToolCalls {
				args, _ := json.Marshal(tc.Function.Arguments)
				toolCalls = append(toolCalls, provider.ToolCall{
					ID:    fmt.Sprintf("call_%d", i),
					Type:  "function",
					Index: i,
					Function: provider.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: string(args),
					},
				})
			}

			finishReason := ""
			if resp.Done {
				finishReason = provider.FinishReasonStop
				if resp.DoneReason == "length" {
					finishReason = provider.FinishReasonLength
				} else if len(toolCalls) > 0 {
					finishReason = provider.FinishReasonToolCalls
				}
			}

			event := provider.StreamEvent{
				Delta: provider.Delta{
					Content:   resp.Message.Content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			}

			select {
			case events <- event:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			case <-done:
				return fmt.Errorf("stream closed")
			}
		})

		if err != nil {
			events <- provider.StreamEvent{Err: err}
		}
	}()

	return provider.NewStreamReader(events, func() { close(done) }), nil
}

func (o *ollama) convertMessages(messages []provider.Message) []api.Message {
	result := make([]api.Message, 0, len(messages))

	for _, msg := range messages {
		apiMsg := api.Message{
			Role:    string(msg.Role),
			Content: msg.Content,
		}

		if len(msg.ToolCalls) > 0 {
			apiMsg.ToolCalls = make([]api.ToolCall, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				var args map[string]any
				json.Unmarshal([]byte(tc.Function.Arguments), &args)
				apiMsg.ToolCalls[i] = api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				}
			}
		}

		result = append(result, apiMsg)
	}

	return result
}

func (o *ollama) convertTools(tools []provider.Tool) []api.Tool {
	result := make([]api.Tool, len(tools))
	for i, t := range tools {
		properties := make(map[string]api.ToolProperty)
		if props, ok := t.Function.Parameters["properties"].(map[string]any); ok {
			for name, prop := range props {
				if p, ok := prop.(map[string]any); ok {
					propType, _ := p["type"].(string)
					propDesc, _ := p["description"].(string)
					properties[name] = api.ToolProperty{
						Type:        api.PropertyType{propType},
						Description: propDesc,
					}
				}
			}
		}

		var required []string
		if req, ok := t.Function.Parameters["required"].([]any); ok {
			for _, r := range req {
				required = append(required, fmt.Sprint(r))
			}
		}

		result[i] = api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Required:   required,
					Properties: properties,
				},
			},
		}
	}
	return result
}

func (o *ollama) toProviderResponse(resp *api.ChatResponse, model string) *provider.ChatResponse {
	var toolCalls []provider.ToolCall
	for i, tc := range resp.Message.ToolCalls {
		args, _ := json.Marshal(tc.Function.Arguments)
		toolCalls = append(toolCalls, provider.ToolCall{
			ID:    fmt.Sprintf("call_%d", i),
			Type:  "function",
			Index: i,
			Function: provider.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: string(args),
			},
		})
	}

	finishReason := provider.FinishReasonStop
	if resp.DoneReason == "length" {
		finishReason = provider.FinishReasonLength
	} else if len(toolCalls) > 0 {
		finishReason = provider.FinishReasonToolCalls
	}

	return &provider.ChatResponse{
		ID:      fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: resp.CreatedAt.Unix(),
		Model:   model,
		Choices: []provider.Choice{
			{
				Index: 0,
				Message: provider.Message{
					Role:      provider.RoleAssistant,
					Content:   resp.Message.Content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: provider.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}
