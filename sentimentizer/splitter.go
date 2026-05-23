package main

import (
	"fmt"
	"strings"
)

// TextSplitter holds the configuration for recursive splitting.
type TextSplitter struct {
	Separators   []string
	ChunkSize    int
	ChunkOverlap int
}

// SplitText is the main entry point for breaking down a document.
func (ts *TextSplitter) SplitText(text string) []string {
	return ts.split(text, ts.Separators)
}

// split handles the recursive breakdown of the text.
func (ts *TextSplitter) split(text string, separators []string) []string {
	// 1. Find the highest priority separator that actually exists in the text
	separator := separators[len(separators)-1] // default to the fallback (usually "")
	var nextSeparators []string

	for i, sep := range separators {
		if sep == "" || strings.Contains(text, sep) {
			separator = sep
			nextSeparators = separators[i+1:]
			break
		}
	}

	// 2. Perform the split
	var splits []string
	if separator == "" {
		// If we've run out of separators, split by individual characters (runes)
		for _, r := range text {
			splits = append(splits, string(r))
		}
	} else {
		splits = strings.Split(text, separator)
	}

	// 3. Recursively split any chunks that are still too large
	var finalSplits []string
	for _, s := range splits {
		if len(s) > ts.ChunkSize && len(nextSeparators) > 0 {
			// Recurse with the remaining, more granular separators
			finalSplits = append(finalSplits, ts.split(s, nextSeparators)...)
		} else if s != "" {
			finalSplits = append(finalSplits, s)
		}
	}

	// 4. Recombine the small pieces into perfectly sized chunks with overlap
	return ts.mergeSplits(finalSplits, separator)
}

// mergeSplits stitches the broken pieces back together up to the ChunkSize.
func (ts *TextSplitter) mergeSplits(splits []string, separator string) []string {
	var chunks []string
	var currentChunk []string
	currentLen := 0

	for _, split := range splits {
		splitLen := len(split)

		// If adding this piece exceeds our chunk size, save the current chunk
		if currentLen+splitLen+len(separator) > ts.ChunkSize && len(currentChunk) > 0 {
			chunks = append(chunks, strings.Join(currentChunk, separator))

			// Slide the window forward for overlap: drop older pieces until we fit the overlap limit
			for len(currentChunk) > 0 && (currentLen > ts.ChunkOverlap || currentLen+splitLen+len(separator) > ts.ChunkSize) {
				currentLen -= len(currentChunk[0])
				if len(currentChunk) > 1 {
					currentLen -= len(separator) // remove the separator length too
				}
				currentChunk = currentChunk[1:]
			}
		}

		currentChunk = append(currentChunk, split)
		currentLen += splitLen
		if len(currentChunk) > 1 {
			currentLen += len(separator)
		}
	}

	// Append whatever is left over as the final chunk
	if len(currentChunk) > 0 {
		chunks = append(chunks, strings.Join(currentChunk, separator))
	}

	return chunks
}

func main() {
	text := `This is the first paragraph. It has some sentences.

This is the second paragraph. It contains more information that we might want to split into smaller chunks. 
We are demonstrating the recursive text splitter in Go.`

	// Configure the splitter
	splitter := &TextSplitter{
		Separators:   []string{"\n\n", "\n", ".", " ", ""}, // Order of priority
		ChunkSize:    65,                                   // Max bytes per chunk
		ChunkOverlap: 15,                                   // Target bytes for overlap
	}

	chunks := splitter.SplitText(text)

	// Print the results
	for i, chunk := range chunks {
		fmt.Printf("Chunk %d (Length: %d):\n%q\n\n", i+1, len(chunk), chunk)
	}
}