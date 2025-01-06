# 📚 AI読書アシスタント: ページごとのPDF知識抽出＆要約ツール

このスクリプトは、PDFの書籍を1ページずつインテリジェントに分析し、知識ポイントを抽出し、指定した間隔で進行的な要約を生成します。各ページを個別に処理しながら、書籍全体の文脈の流れを維持します。OpenAI GPTとGoogle Gemini Pro APIの両方をサポートしています！

### 機能

- 📚 PDFブックの自動分析と知識抽出
- 🤖 AIによるコンテンツ理解と要約（OpenAI GPTまたはGoogle Gemini Pro）
- 📊 一定間隔での進捗要約
- 💾 知識ベースの永続的な保存
- 📝 Markdown形式の要約
- 🎨 見やすい色分けされたターミナル出力
- 🔄 既存の知識ベースからの再開機能
- ⚙️ 設定可能な分析間隔とテストモード
- 🚫 スマートなコンテンツフィルタリング（目次、索引ページなどをスキップ）
- 📂 整理された出力ディレクトリ構造
- 🌏 日本語コンテンツの完全サポート

## ❤️ サポートと400以上のAIプロジェクト入手

これは400以上の魅力的なプロジェクトコレクションの1つです！[Patreonでサポート](https://www.patreon.com/c/echohive42/membership)して以下を入手できます：

- 🎯 400以上のAIプロジェクトへのアクセス（日々増加中！）
- 📥 完全なソースコードと詳細な説明
- 📚 1000x Cursorコース
- 🎓 ライブコーディングセッションとAMA
- 💬 1対1のコンサルテーション（上位ティア）
- 🎁 AIツールとプラットフォームの限定割引（最大$180相当）

## 使い方

1. **セットアップ**
   ```bash
   # リポジトリのクローン
   git clone [repository-url]
   cd [repository-name]

   # 必要なパッケージのインストール
   pip install -r requirements.txt
   ```

2. **設定**
   - `.env.example`を参考に`.env`ファイルを作成
   - APIキーを設定（OPENAI_API_KEYまたはGEMINI_API_KEY）
   - PDFファイルをプロジェクトのルートディレクトリに配置

3. **実行**
   OpenAI GPTを使用する場合：
   ```bash
   python read_books.py --pdf 本の名前.pdf
   ```

   Google Gemini Proを使用する場合：
   ```bash
   # 英語版
   python pdf_reader_gemini.py --pdf book.pdf
   # 日本語版
   python pdf_reader_gemini_ja.py --pdf 本の名前.pdf
   ```

4. **出力**
   スクリプトは以下を生成します：
   - `book_analysis/knowledge_bases/`: 抽出された知識を含むJSONファイル
   - `book_analysis/summaries/`: 中間要約と最終要約のMarkdownファイル
   - `book_analysis/pdfs/`: PDFファイルのコピー

5. **カスタマイズオプション**
   - `ANALYSIS_INTERVAL`を`None`に設定すると中間要約をスキップ
   - `TEST_PAGES`を`None`に設定すると書籍全体を処理
   - 分析にOpenAI GPTとGemini Proを選択可能

### 設定項目

- `PDF_NAME`: 分析対象のPDFファイル名
- `BASE_DIR`: 分析のベースディレクトリ
- `PDF_DIR`: PDFファイルの保存ディレクトリ
- `KNOWLEDGE_DIR`: 知識ベースの保存ディレクトリ
- `SUMMARIES_DIR`: 要約の保存ディレクトリ
- `PDF_PATH`: PDFファイルへのフルパス
- `OUTPUT_PATH`: 知識ベースJSONファイルのパス
- `ANALYSIS_INTERVAL`: 中間分析を生成するページ間隔。`None`で中間分析をスキップ
- `MODEL`: ページ処理に使用するモデル
- `ANALYSIS_MODEL`: 分析生成に使用するモデル
- `TEST_PAGES`: テスト用の処理ページ数。`None`で書籍全体を処理

### API選択の考慮点

#### OpenAI GPT
- より確立され広くテストされている
- 一般的により一貫性のある結果を提供
- コストは高いが品質が良い可能性がある

#### Google Gemini Pro
- 競争力のある機能を持つ新しい選択肢
- よりコスト効率が良い
- 急速に成長・改善中
- 応答時間が潜在的に速い
- 日本語の処理が特に優れている

### 仕組み

1. **セットアップ**: 必要なディレクトリを設定し、PDFファイルが正しい場所にあることを確認
2. **知識ベースの読み込み**: 既存の知識ベースがあれば読み込み
3. **ページ処理**: PDFの各ページを処理し、知識ポイントを抽出して知識ベースを更新
4. **要約生成**: `ANALYSIS_INTERVAL`に基づいて中間要約を生成し、全ページ処理後に最終要約を生成
5. **結果の保存**: 知識ベースと要約を各ファイルに保存

### コマンドライン例

OpenAI GPTを使用：
```bash
python read_books.py --pdf "孫子の兵法.pdf" --interval 5
```

Gemini Pro（日本語版）を使用：
```bash
python pdf_reader_gemini_ja.py --pdf "孫子の兵法.pdf" --interval 5
```

### 日本語PDFに関する注意点

1. **文字コード**
   - PDFファイルはUTF-8エンコーディングを推奨
   - 特殊な文字が含まれる場合は正しく処理されない可能性あり

2. **レイアウト**
   - 縦書きPDFは正しく処理されない可能性あり
   - 複雑なレイアウトの場合、テキスト抽出が不完全な場合あり

3. **OCR**
   - スキャンされたPDFの場合、事前にOCR処理が必要
   - 画像として埋め込まれたテキストは処理不可

4. **最適なパフォーマンス**
   - テキストベースのPDFを使用
   - シンプルなレイアウトの文書を推奨
   - 必要に応じて事前にPDF最適化を実施