# 📚 AI reads books: Page-by-Page PDF Knowledge Extractor & Summarizer

The script performs an intelligent page-by-page analysis of PDF books, methodically extracting knowledge points and generating progressive summaries at specified intervals. It processes each page individually, allowing for detailed content understanding while maintaining the contextual flow of the book. Now with support for both OpenAI GPT and Google's Gemini Pro API!

### Features

- 📚 Automated PDF book analysis and knowledge extraction
- 🤖 AI-powered content understanding and summarization (OpenAI GPT or Google Gemini Pro)
- 📊 Interval-based progress summaries
- 💾 Persistent knowledge base storage
- 📝 Markdown-formatted summaries
- 🎨 Color-coded terminal output for better visibility
- 🔄 Resume capability with existing knowledge base
- ⚙️ Configurable analysis intervals and test modes
- 🚫 Smart content filtering (skips TOC, index pages, etc.)
- 📂 Organized directory structure for outputs
- 🔄 Choice between OpenAI GPT and Google Gemini Pro APIs

## ❤️Join my AI Community & Get 400+ AI Projects & 1000x Cursor Course

This is one of 400+ fascinating projects in my collection! [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools & platforms (up to $180 value)

## How to Use

1. **Setup**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd [repository-name]

   # Install requirements
   pip install -r requirements.txt
   ```

2. **Configure**
   - Create a `.env` file based on `.env.example`
   - Add your API key (either OPENAI_API_KEY or GEMINI_API_KEY)
   - Place your PDF file in the project root directory

3. **Run**
   For OpenAI GPT:
   ```bash
   python read_books.py --pdf your_book.pdf
   ```

   For Google Gemini Pro:
   ```bash
   python pdf_reader_gemini.py --pdf your_book.pdf
   ```

4. **Output**
   The script will generate:
   - `book_analysis/knowledge_bases/`: JSON files containing extracted knowledge
   - `book_analysis/summaries/`: Markdown files with interval and final summaries
   - `book_analysis/pdfs/`: Copy of your PDF file

5. **Customization Options**
   - Set `ANALYSIS_INTERVAL = None` to skip interval summaries
   - Set `TEST_PAGES = None` to process entire book
   - Choose between OpenAI GPT and Gemini Pro for analysis

### Configuration Constants

- `PDF_NAME`: The name of the PDF file to be analyzed.
- `BASE_DIR`: The base directory for the analysis.
- `PDF_DIR`: Directory where the PDF file is stored.
- `KNOWLEDGE_DIR`: Directory where the knowledge base will be saved.
- `SUMMARIES_DIR`: Directory where the summaries will be saved.
- `PDF_PATH`: Full path to the PDF file.
- `OUTPUT_PATH`: Path to the knowledge base JSON file.
- `ANALYSIS_INTERVAL`: Number of pages after which an interval analysis is generated. Set to `None` to skip interval analyses.
- `MODEL`: The model used for processing pages.
- `ANALYSIS_MODEL`: The model used for generating analyses.
- `TEST_PAGES`: Number of pages to process for testing. Set to `None` to process the entire book.

### Classes and Functions

#### `PageContent` Class

A Pydantic model that represents the structure of the response from the AI API for page content analysis. It has two fields:

- `has_content`: A boolean indicating if the page has relevant content.
- `knowledge`: A list of knowledge points extracted from the page.

#### `load_or_create_knowledge_base() -> Dict[str, Any]`

Loads the existing knowledge base from the JSON file if it exists. If not, it returns an empty dictionary.

#### `save_knowledge_base(knowledge_base: list[str])`

Saves the knowledge base to a JSON file. It prints a message indicating the number of items saved.

#### `process_page(client, page_text: str, current_knowledge: list[str], page_num: int) -> list[str]`

Processes a single page of the PDF. It sends the page text to the AI API for analysis and updates the knowledge base with the extracted knowledge points. It also saves the updated knowledge base to a JSON file.

#### `load_existing_knowledge() -> list[str]`

Loads the existing knowledge base from the JSON file if it exists. If not, it returns an empty list.

#### `analyze_knowledge_base(client, knowledge_base: list[str]) -> str`

Generates a comprehensive summary of the entire knowledge base using the AI API. It returns the summary in markdown format.

#### `setup_directories()`

Sets up the necessary directories for the analysis. It clears any previously generated files and ensures the PDF file is in the correct location.

#### `save_summary(summary: str, is_final: bool = False)`

Saves the generated summary to a markdown file. It creates a file with a proper naming convention based on whether it is a final or interval summary.

#### `print_instructions()`

Prints instructions for using the script. It explains the configuration options and how to run the script.

#### `main()`

The main function that orchestrates the entire process. It sets up directories, loads the knowledge base, processes each page of the PDF, generates interval and final summaries, and saves them.

### How It Works

1. **Setup**: The script sets up the necessary directories and ensures the PDF file is in the correct location.
2. **Load Knowledge Base**: It loads the existing knowledge base if it exists.
3. **Process Pages**: It processes each page of the PDF, extracting knowledge points and updating the knowledge base.
4. **Generate Summaries**: It generates interval summaries based on the `ANALYSIS_INTERVAL` and a final summary after processing all pages.
5. **Save Results**: It saves the knowledge base and summaries to their respective files.

### API Choice Considerations

#### OpenAI GPT
- More established and widely tested
- Generally provides more consistent results
- Higher costs but potentially better quality

#### Google Gemini Pro
- Newer alternative with competitive capabilities
- More cost-effective option
- Growing and improving rapidly
- Potentially faster response times

### Example Usage

Using OpenAI GPT:
```bash
python read_books.py --pdf "The Art of War.pdf" --interval 5
```

Using Gemini Pro:
```bash
python pdf_reader_gemini.py --pdf "The Art of War.pdf" --interval 5
```
