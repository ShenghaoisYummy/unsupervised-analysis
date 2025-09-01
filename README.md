# Interview Analysis System

An automated system for analyzing interview data using LightRAG knowledge base and OpenAI GPT-4 for unsupervised topic and theme clustering.

## Features

- **Document Parsing**: Convert Word documents (.docx) to clean markdown format
- **Knowledge Base**: Create searchable knowledge base using LightRAG with OpenAI embeddings
- **Two-Stage Clustering**:
  - **Stage 1**: Within-article topic clustering for each interview
  - **Stage 2**: Cross-article theme clustering to group related interviews
- **AI Analysis**: GPT-4 powered topic extraction and theme identification
- **Results Export**: Structured JSON outputs and analysis reports

## Quick Start

### 1. Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Copy environment template and add your OpenAI API key
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

#### Alternative setup with Make:
```bash
make setup  # Install uv + sync dependencies + copy .env template
```

### 2. Prepare Your Data

Place your interview files in the appropriate directory:

- **Word documents**: Put `.docx` files in `./documents/`
- **Markdown files**: Put `.md` files directly in `./interviews/`

### 3. Run Analysis

```bash
# Run complete analysis pipeline
uv run src/main_analyzer.py

# Or run specific stages
uv run src/main_analyzer.py --stage convert      # Convert documents only
uv run src/main_analyzer.py --stage knowledge   # Build knowledge base only
uv run src/main_analyzer.py --stage stage1      # Run Stage 1 analysis only
uv run src/main_analyzer.py --stage stage2      # Run Stage 2 analysis only

# Skip document conversion if you already have markdown files
uv run src/main_analyzer.py --skip-conversion
```

#### Alternative with Make:
```bash
make run                    # Run complete pipeline
make run-convert           # Convert documents only
make run-stage1            # Run Stage 1 only
make run-stage2            # Run Stage 2 only
```

## Project Structure

```
unsupervised-analysis/
├── documents/              # Input Word documents (.docx files)
├── interviews/            # Converted markdown files
├── src/
│   ├── config.py         # Configuration management
│   ├── docx_parser.py    # Word to Markdown converter
│   ├── lightrag_setup.py # LightRAG knowledge base
│   ├── stage1_clustering.py # Within-article topic clustering
│   ├── stage2_clustering.py # Cross-article theme clustering
│   └── main_analyzer.py  # Main orchestrator script
├── results/              # Analysis outputs (JSON files)
├── rag_storage/         # LightRAG knowledge base files
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Lockfile for reproducible installs
├── .python-version     # Python version specification
├── Makefile            # Convenient development commands
├── .env.template       # Environment variables template
├── requirements.txt    # Legacy pip requirements (optional)
└── README.md
```

## Analysis Pipeline

### Stage 1: Within-Article Topic Clustering
- Extracts key topics from each interview using GPT-4
- Clusters similar topics within each article
- Identifies topic importance and relationships
- Output: `results/stage1_within_article_topics.json`

### Stage 2: Cross-Article Theme Clustering
- Extracts high-level themes from each article's topics
- Clusters articles by thematic similarity using K-means
- Analyzes cluster patterns with GPT-4
- Uses LightRAG for cross-article relationship discovery
- Output: `results/stage2_cross_article_themes.json`

## Configuration

Edit `.env` file to configure:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
RAG_WORKING_DIR=./rag_storage
MAX_TOPICS_PER_ARTICLE=10
CLUSTERING_THRESHOLD=0.7
```

## Output Files

### Stage 1 Results
```json
{
  "stage": 1,
  "description": "Within-article topic clustering",
  "total_articles": 70,
  "articles": [
    {
      "filename": "interview_01.md",
      "topics_count": 8,
      "clustered_count": 5,
      "clustered_topics": [
        {
          "name": "Career Development",
          "description": "Discussion about professional growth",
          "importance": 9,
          "key_phrases": ["promotion", "skills", "advancement"],
          "supporting_quotes": ["..."]
        }
      ]
    }
  ]
}
```

### Stage 2 Results
```json
{
  "stage": 2,
  "description": "Cross-article theme clustering",
  "n_clusters": 8,
  "silhouette_score": 0.75,
  "clusters": [
    {
      "cluster_number": 0,
      "size": 12,
      "analysis": {
        "cluster_name": "Career Transitions",
        "main_theme": "Professional development and career changes",
        "common_patterns": ["job changes", "skill development", "mentorship"]
      },
      "articles": [...]
    }
  ]
}
```

## Requirements

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key with GPT-4 access
- Dependencies managed through `pyproject.toml`

## Development

### Using uv for development:

```bash
# Install with development dependencies
make dev-install
# or
uv sync --extra dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Type checking
make check
```

## Advanced Usage

### Individual Component Usage

```python
# Run in uv environment
# uv run python -c "
from src.docx_parser import DocxToMarkdownParser
from src.lightrag_setup import LightRAGKnowledgeBase
from src.stage1_clustering import WithinArticleTopicClustering
from src.stage2_clustering import CrossArticleThemeClustering

# Convert documents
parser = DocxToMarkdownParser()
converted_files = parser.convert_batch()

# Build knowledge base
kb = LightRAGKnowledgeBase()
await kb.build_knowledge_base()

# Run clustering stages
stage1 = WithinArticleTopicClustering()
results1 = await stage1.process_all_articles()

stage2 = CrossArticleThemeClustering()
results2 = await stage2.process_stage2_clustering()
# "
```

### Using the CLI script:
```bash
# The package installs a CLI script
uv run interview-analyzer --help
uv run interview-analyzer --stage stage1
```

### Query Knowledge Base

```python
from src.lightrag_setup import LightRAGKnowledgeBase

kb = LightRAGKnowledgeBase()
await kb.initialize_rag()

# Query the knowledge base
response = await kb.query_knowledge_base(
    "What are the main challenges discussed across interviews?",
    mode="hybrid"
)
```

## Troubleshooting

1. **API Rate Limits**: The system includes automatic delays between API calls
2. **Memory Issues**: For large datasets, consider processing in smaller batches
3. **Token Limits**: Long articles are automatically truncated to stay within GPT-4 limits
4. **JSON Parsing**: The system handles malformed GPT responses gracefully

## License

MIT License - feel free to modify and use for your research projects.
