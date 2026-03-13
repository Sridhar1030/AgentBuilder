/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dsp

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// Client talks to the Data Science Pipelines v2beta1 REST API.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient builds a DSP client for the given namespace.
// It uses the in-cluster service URL and skips TLS verification
// (the DSP service uses an internal self-signed cert on port 8443).
func NewClient(namespace string) *Client {
	baseURL := fmt.Sprintf("https://ds-pipeline-dspa.%s.svc.cluster.local:8443", namespace)
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec
			},
		},
	}
}

func (c *Client) bearerToken() (string, error) {
	token, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/token")
	if err != nil {
		return "", fmt.Errorf("reading SA token: %w", err)
	}
	return strings.TrimSpace(string(token)), nil
}

func (c *Client) doRequest(method, path string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshalling request body: %w", err)
		}
		reqBody = bytes.NewReader(b)
	}

	req, err := http.NewRequest(method, c.BaseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	token, err := c.bearerToken()
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("DSP API %s %s returned %d: %s", method, path, resp.StatusCode, string(data))
	}

	return data, nil
}

// FindPipelineByName looks up a pipeline by display name and returns its ID.
func (c *Client) FindPipelineByName(name string) (string, error) {
	filter := fmt.Sprintf(`{"predicates":[{"key":"name","operation":"EQUALS","string_value":"%s"}]}`, name)
	path := "/apis/v2beta1/pipelines?filter=" + url.QueryEscape(filter)

	data, err := c.doRequest(http.MethodGet, path, nil)
	if err != nil {
		return "", fmt.Errorf("listing pipelines: %w", err)
	}

	var result struct {
		Pipelines []struct {
			PipelineID string `json:"pipeline_id"`
		} `json:"pipelines"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("parsing pipeline list: %w", err)
	}
	if len(result.Pipelines) == 0 {
		return "", fmt.Errorf("pipeline %q not found", name)
	}
	return result.Pipelines[0].PipelineID, nil
}

// RunParams holds the pipeline runtime parameters.
type RunParams struct {
	ModelVersion     string
	GroqAPIKey       string
	GroqModel        string
	NumEpochs        int
	MinGoldThreshold int
}

// CreateRun submits a new pipeline run and returns the run ID.
func (c *Client) CreateRun(displayName, pipelineID string, params RunParams) (string, error) {
	body := map[string]interface{}{
		"display_name": displayName,
		"pipeline_version_reference": map[string]interface{}{
			"pipeline_id": pipelineID,
		},
		"runtime_config": map[string]interface{}{
			"parameters": map[string]interface{}{
				"model_version":      params.ModelVersion,
				"groq_api_key":       params.GroqAPIKey,
				"groq_model":         params.GroqModel,
				"num_epochs":         params.NumEpochs,
				"min_gold_threshold": params.MinGoldThreshold,
			},
		},
	}

	data, err := c.doRequest(http.MethodPost, "/apis/v2beta1/runs", body)
	if err != nil {
		return "", fmt.Errorf("creating run: %w", err)
	}

	var result struct {
		RunID string `json:"run_id"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("parsing create-run response: %w", err)
	}
	if result.RunID == "" {
		return "", fmt.Errorf("empty run_id in response: %s", string(data))
	}
	return result.RunID, nil
}

// RunState represents the current state of a pipeline run.
type RunState struct {
	State   string // PENDING, RUNNING, SUCCEEDED, FAILED, etc.
	Message string
}

// GetRun fetches the current state of a pipeline run.
func (c *Client) GetRun(runID string) (RunState, error) {
	data, err := c.doRequest(http.MethodGet, "/apis/v2beta1/runs/"+runID, nil)
	if err != nil {
		return RunState{}, fmt.Errorf("getting run %s: %w", runID, err)
	}

	var result struct {
		State       string                    `json:"state"`
		Error       *struct{ Message string } `json:"error"`
		DisplayName string                    `json:"display_name"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return RunState{}, fmt.Errorf("parsing run response: %w", err)
	}

	rs := RunState{State: result.State}
	if result.Error != nil {
		rs.Message = result.Error.Message
	}
	return rs, nil
}
