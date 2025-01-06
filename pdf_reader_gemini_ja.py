import os
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import json

# 環境変数の読み込み
load_dotenv()

# Gemini APIの設定
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY environment variable is not set')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        print(f'PDFの総ページ数: {total_pages}')
        
        text_content = []
        for page in range(total_pages):
            text = pdf_reader.pages[page].extract_text()
            text_content.append(text)
        
        return text_content

def analyze_page(page_text, page_number):
    prompt = f"""以下の本のページ{page_number}を分析し、重要なポイントと洞察を抽出してください：

{page_text}

このページで議論されている主要なポイントと重要な概念について、簡潔な要約を提供してください。"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f'ページ{page_number}の分析中にエラーが発生しました: {str(e)}')
        return f'ページ{page_number}の分析中にエラーが発生しました'

def main(pdf_path, output_path='analysis_results.json', summary_interval=10):
    # PDFからテキストを抽出
    pages_content = extract_text_from_pdf(pdf_path)
    
    # 各ページを分析
    analysis_results = []
    progressive_summaries = []
    
    print('\nページを分析中...')
    for i, page_text in enumerate(tqdm(pages_content)):
        # 個別ページの分析
        page_analysis = analyze_page(page_text, i + 1)
        analysis_results.append({
            'page_number': i + 1,
            'analysis': page_analysis
        })
        
        # 指定された間隔で進行的な要約を生成
        if (i + 1) % summary_interval == 0:
            pages_to_summarize = pages_content[i - summary_interval + 1:i + 1]
            combined_text = '\n'.join(pages_to_summarize)
            
            summary_prompt = f"""以下の節（{i - summary_interval + 2}ページから{i + 1}ページ）の包括的な要約を提供してください：

{combined_text}

この節で扱われている主要なテーマ、概念、展開について簡潔な要約を提供してください。"""
            
            try:
                summary_response = model.generate_content(summary_prompt)
                progressive_summaries.append({
                    'pages': f'{i - summary_interval + 2}-{i + 1}',
                    'summary': summary_response.text
                })
            except Exception as e:
                print(f'ページ{i - summary_interval + 2}-{i + 1}の要約生成中にエラーが発生しました: {str(e)}')
    
    # 結果を保存
    results = {
        'page_analyses': analysis_results,
        'progressive_summaries': progressive_summaries
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\n分析が完了しました！結果は{output_path}に保存されました')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PDFブック分析ツール（Gemini API使用）')
    parser.add_argument('pdf_path', help='PDFファイルのパス')
    parser.add_argument('--output', default='analysis_results.json', help='出力JSONファイルのパス')
    parser.add_argument('--summary-interval', type=int, default=10, help='進行的な要約を生成するページ間隔')
    
    args = parser.parse_args()
    main(args.pdf_path, args.output, args.summary_interval)
