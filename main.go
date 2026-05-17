package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

type SentimentRequest struct {
	Text string `json:"text"`
}

type SentimentResponse struct {
	Model      string             `json:"model"`
	Label      string             `json:"label"`
	Scores     map[string]float64 `json:"scores"`
	Error      string             `json:"error,omitempty"`
}

func main() {
	host := flag.String("host", "http://localhost:8000", "Ray Serve endpoint")
	text := flag.String("text", "", "Text to analyze")
	timeout := flag.Duration("timeout", 30*time.Second, "Request timeout")
	raw := flag.Bool("raw", false, "Output raw JSON")
	flag.Parse()

	// If no -text flag, check for positional args or stdin
	input := *text
	if input == "" && flag.NArg() > 0 {
		input = strings.Join(flag.Args(), " ")
	}
	if input == "" {
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			stdinData, _ := io.ReadAll(os.Stdin)
			input = strings.TrimSpace(string(stdinData))
		}
	}
	if input == "" {
		fmt.Fprintf(os.Stderr, "Usage: sentiment -text \"some text to analyze\"\n")
		fmt.Fprintf(os.Stderr, "       echo \"some text\" | sentiment\n")
		fmt.Fprintf(os.Stderr, "       sentiment \"some text as positional arg\"\n")
		os.Exit(1)
	}

	// Build request
	reqBody, _ := json.Marshal(SentimentRequest{Text: input})

	client := &http.Client{Timeout: *timeout}
	start := time.Now()

	resp, err := client.Post(*host, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)

	if resp.StatusCode >= 400 {
		fmt.Fprintf(os.Stderr, "Server error %d: %s\n", resp.StatusCode, string(respBody))
		os.Exit(1)
	}

	// Raw JSON output
	if *raw {
		var pretty bytes.Buffer
		json.Indent(&pretty, respBody, "", "  ")
		fmt.Println(pretty.String())
		return
	}

	// Parse and display nicely
	var result SentimentResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing response: %v\n", err)
		fmt.Fprintf(os.Stderr, "Raw: %s\n", string(respBody))
		os.Exit(1)
	}

	if result.Error != "" {
		fmt.Fprintf(os.Stderr, "Server error: %s\n", result.Error)
		os.Exit(1)
	}

	// Colorized output
	emoji := "👎"
	color := "\033[31m" // red
	if result.Label == "positive" {
		emoji = "👍"
		color = "\033[32m" // green
	} else if result.Label == "neutral" {
		emoji = "😐"
		color = "\033[33m" // yellow
	}
	reset := "\033[0m"

	fmt.Printf("Text:       %s\n", input)
	fmt.Printf("Prediction: %s%s%s %s\n", color, result.Label, reset, emoji)
	fmt.Printf("Model:      %s\n", result.Model)
	// Print scores sorted by label name
	labels := make([]string, 0, len(result.Scores))
	for label := range result.Scores {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		fmt.Printf("  %-8s: %.4f\n", label, result.Scores[label])
	}
	fmt.Printf("Latency:    %s\n", elapsed.Round(time.Millisecond))
}