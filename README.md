# 📚 AI reads books: Page-by-Page PDF Knowledge Extractor & Summarizer

The `read_books.py` script performs an intelligent page-by-page analysis of PDF books, methodically extracting knowledge points and generating progressive summaries at specified intervals. It processes each page individually, allowing for detailed content understanding while maintaining the contextual flow of the book. Below is a detailed explanation of how the script works:

### Features

- 📚 Automated PDF book analysis and knowledge extraction
- 🤖 AI-powered content understanding and summarization
- 📊 Interval-based progress summaries
- 💾 Persistent knowledge base storage
- 📝 Markdown-formatted summaries
- 🎨 Color-coded terminal output for better visibility
- 🔄 Resume capability with existing knowledge base
- ⚙️ Configurable analysis intervals and test modes
- 🚫 Smart content filtering (skips TOC, index pages, etc.)
- 📂 Organized directory structure for outputs

## ❤️ Go deeper: Get Amplified + weekly 1000x LAB meetings

This repo is one small doorway into the larger AI-building practice I share with patrons.

- ❤️ [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get the full project collection, source code, explanations, and ongoing AI-building material.
- 🎥 [Get Amplified](https://www.patreon.com/collection/1845761) is my 55-video-post series for thinking fast, building faster, and speed-running your creativity. Learn to use Codex, Claude Code, Cursor, and other AI tools effectively and creatively. Get amplified by becoming amplifiable, with more chapters on the way in quick succession.
- 🧠 [1000x LAB for Architect+ tiers](https://www.patreon.com/collection/759209) is the patron meeting archive: 82 focused 1000x meetings so far, with a new one added every week. These sessions go behind the scenes on real builds, agent workflows, creative tooling, and the decisions that turn experiments into finished, usable systems.
- 🚀 Patrons get the deeper context around projects like this: source code, walkthroughs, implementation notes, and a steady stream of examples for turning AI ideas into working products.

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
   - Place your PDF file in the project root directory
   - Open `read_books.py` and update the `PDF_NAME` constant with your PDF filename
   - (Optional) Adjust other constants like `ANALYSIS_INTERVAL` or `TEST_PAGES`

3. **Run**
   ```bash
   python read_books.py
   ```

4. **Output**
   The script will generate:
   - `book_analysis/knowledge_bases/`: JSON files containing extracted knowledge
   - `book_analysis/summaries/`: Markdown files with interval and final summaries
   - `book_analysis/pdfs/`: Copy of your PDF file

5. **Customization Options**
   - Set `ANALYSIS_INTERVAL = None` to skip interval summaries
   - Set `TEST_PAGES = None` to process entire book
   - Adjust `MODEL` and `ANALYSIS_MODEL` for different AI models

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

A Pydantic model that represents the structure of the response from the OpenAI API for page content analysis. It has two fields:

- `has_content`: A boolean indicating if the page has relevant content.
- `knowledge`: A list of knowledge points extracted from the page.

#### `load_or_create_knowledge_base() -> Dict[str, Any]`

Loads the existing knowledge base from the JSON file if it exists. If not, it returns an empty dictionary.

#### `save_knowledge_base(knowledge_base: list[str])`

Saves the knowledge base to a JSON file. It prints a message indicating the number of items saved.

#### `process_page(client: OpenAI, page_text: str, current_knowledge: list[str], page_num: int) -> list[str]`

Processes a single page of the PDF. It sends the page text to the OpenAI API for analysis and updates the knowledge base with the extracted knowledge points. It also saves the updated knowledge base to a JSON file.

#### `load_existing_knowledge() -> list[str]`

Loads the existing knowledge base from the JSON file if it exists. If not, it returns an empty list.

#### `analyze_knowledge_base(client: OpenAI, knowledge_base: list[str]) -> str`

Generates a comprehensive summary of the entire knowledge base using the OpenAI API. It returns the summary in markdown format.

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

### Running the Script

1. Place your PDF in the same directory as the script.
2. Update the `PDF_NAME` constant with your PDF filename.
3. Run the script. It will process the book, extract knowledge points, and generate summaries.

### Example Usage
