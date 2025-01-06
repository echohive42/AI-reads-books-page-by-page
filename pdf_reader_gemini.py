import os
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import json

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY environment variable is not set')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        print(f'Total pages in PDF: {total_pages}')
        
        text_content = []
        for page in range(total_pages):
            text = pdf_reader.pages[page].extract_text()
            text_content.append(text)
        
        return text_content

def analyze_page(page_text, page_number):
    prompt = f"""Analyze the following page {page_number} from a book and extract key points and insights:

{page_text}

Provide a concise summary of the main points and any important concepts discussed on this page."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f'Error analyzing page {page_number}: {str(e)}')
        return f'Error analyzing page {page_number}'

def main(pdf_path, output_path='analysis_results.json', summary_interval=10):
    # Extract text from PDF
    pages_content = extract_text_from_pdf(pdf_path)
    
    # Analyze each page
    analysis_results = []
    progressive_summaries = []
    
    print('\nAnalyzing pages...')
    for i, page_text in enumerate(tqdm(pages_content)):
        # Analyze individual page
        page_analysis = analyze_page(page_text, i + 1)
        analysis_results.append({
            'page_number': i + 1,
            'analysis': page_analysis
        })
        
        # Generate progressive summary at intervals
        if (i + 1) % summary_interval == 0:
            pages_to_summarize = pages_content[i - summary_interval + 1:i + 1]
            combined_text = '\n'.join(pages_to_summarize)
            
            summary_prompt = f"""Provide a comprehensive summary of the following section (pages {i - summary_interval + 2}-{i + 1}):

{combined_text}

Provide a concise summary that captures the main themes, concepts, and developments in this section."""
            
            try:
                summary_response = model.generate_content(summary_prompt)
                progressive_summaries.append({
                    'pages': f'{i - summary_interval + 2}-{i + 1}',
                    'summary': summary_response.text
                })
            except Exception as e:
                print(f'Error generating summary for pages {i - summary_interval + 2}-{i + 1}: {str(e)}')
    
    # Save results
    results = {
        'page_analyses': analysis_results,
        'progressive_summaries': progressive_summaries
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\nAnalysis complete! Results saved to {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PDF Book Analyzer using Gemini API')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', default='analysis_results.json', help='Output JSON file path')
    parser.add_argument('--summary-interval', type=int, default=10, help='Page interval for progressive summaries')
    
    args = parser.parse_args()
    main(args.pdf_path, args.output, args.summary_interval)
