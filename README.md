# AI Reads Books: Page-by-Page PDF Knowledge Extractor & Summarizer

This project provides an intelligent way to analyze PDF books page by page, extracting key knowledge points and generating progressive summaries at specified intervals. The script now supports both OpenAI's GPT models and Google's Gemini Pro model.

## Features

- Page-by-page PDF text extraction
- Detailed analysis of each page's content
- Progressive summaries at specified intervals
- Support for both OpenAI GPT and Google Gemini Pro
- Results saved in JSON format for easy processing

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and add your API key:
   - For OpenAI GPT: Use `pdf_reader.py` and set `OPENAI_API_KEY`
   - For Gemini Pro: Use `pdf_reader_gemini.py` and set `GEMINI_API_KEY`

## Usage

Using Gemini Pro:
```bash
python pdf_reader_gemini.py path/to/your/book.pdf [--output results.json] [--summary-interval 10]
```

Using OpenAI GPT:
```bash
python pdf_reader.py path/to/your/book.pdf [--output results.json] [--summary-interval 10]
```

### Parameters

- `pdf_path`: Path to your PDF file (required)
- `--output`: Path for the output JSON file (default: analysis_results.json)
- `--summary-interval`: Number of pages after which to generate a progressive summary (default: 10)

## Output

The script generates a JSON file containing:
- Individual page analyses
- Progressive summaries at specified intervals

## License

MIT
