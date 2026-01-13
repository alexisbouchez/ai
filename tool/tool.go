package tool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/palmframe/palm/provider"
)

type Handler func(ctx context.Context, args Args) (string, error)

type Tool struct {
	name        string
	description string
	params      []*ParamBuilder
	handler     Handler
}

func New(name string) *Tool {
	return &Tool{name: name}
}

func (t *Tool) Description(desc string) *Tool {
	t.description = desc
	return t
}

func (t *Tool) Input(params ...*ParamBuilder) *Tool {
	t.params = params
	return t
}

func (t *Tool) Execute(h Handler) *Tool {
	t.handler = h
	return t
}

func (t *Tool) Name() string {
	return t.name
}

func (t *Tool) Run(ctx context.Context, argsJSON string) (string, error) {
	if t.handler == nil {
		return "", fmt.Errorf("no handler defined for tool %q", t.name)
	}

	var raw map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &raw); err != nil {
		return "", fmt.Errorf("failed to parse arguments: %w", err)
	}

	return t.handler(ctx, Args(raw))
}

func (t *Tool) ToProvider() provider.Tool {
	properties := make(map[string]any)
	var required []string

	for _, p := range t.params {
		prop := map[string]any{
			"type": p.typ,
		}
		if p.description != "" {
			prop["description"] = p.description
		}
		if len(p.enum) > 0 {
			prop["enum"] = p.enum
		}
		if p.items != "" {
			prop["items"] = map[string]any{"type": p.items}
		}
		properties[p.name] = prop
		if p.required {
			required = append(required, p.name)
		}
	}

	return provider.Tool{
		Type: "function",
		Function: provider.Function{
			Name:        t.name,
			Description: t.description,
			Parameters: map[string]any{
				"type":       "object",
				"properties": properties,
				"required":   required,
			},
		},
	}
}

type ParamBuilder struct {
	name        string
	typ         string
	description string
	required    bool
	enum        []string
	items       string
}

func Param(name string) *ParamBuilder {
	return &ParamBuilder{name: name}
}

func (p *ParamBuilder) String() *ParamBuilder {
	p.typ = "string"
	return p
}

func (p *ParamBuilder) Integer() *ParamBuilder {
	p.typ = "integer"
	return p
}

func (p *ParamBuilder) Number() *ParamBuilder {
	p.typ = "number"
	return p
}

func (p *ParamBuilder) Boolean() *ParamBuilder {
	p.typ = "boolean"
	return p
}

func (p *ParamBuilder) Enum(values ...string) *ParamBuilder {
	p.typ = "string"
	p.enum = values
	return p
}

func (p *ParamBuilder) Array(itemType string) *ParamBuilder {
	p.typ = "array"
	p.items = itemType
	return p
}

func (p *ParamBuilder) Object() *ParamBuilder {
	p.typ = "object"
	return p
}

func (p *ParamBuilder) Required() *ParamBuilder {
	p.required = true
	return p
}

func (p *ParamBuilder) Desc(description string) *ParamBuilder {
	p.description = description
	return p
}

type Args map[string]any

func (a Args) String(key string) string {
	if v, ok := a[key].(string); ok {
		return v
	}
	return ""
}

func (a Args) Int(key string) int {
	switch v := a[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	}
	return 0
}

func (a Args) Float(key string) float64 {
	if v, ok := a[key].(float64); ok {
		return v
	}
	return 0
}

func (a Args) Bool(key string) bool {
	if v, ok := a[key].(bool); ok {
		return v
	}
	return false
}

func (a Args) Strings(key string) []string {
	if v, ok := a[key].([]any); ok {
		result := make([]string, len(v))
		for i, item := range v {
			result[i], _ = item.(string)
		}
		return result
	}
	return nil
}

func (a Args) Object(key string) Args {
	if v, ok := a[key].(map[string]any); ok {
		return Args(v)
	}
	return nil
}

func (a Args) Has(key string) bool {
	_, ok := a[key]
	return ok
}

func (a Args) Raw(key string) any {
	return a[key]
}

func ToProviderTools(tools ...*Tool) []provider.Tool {
	result := make([]provider.Tool, len(tools))
	for i, t := range tools {
		result[i] = t.ToProvider()
	}
	return result
}
