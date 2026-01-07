# Retrieval Augmented Generation (RAG) Module

Modulo completo per RAG Q&A su normative legali (GDPR e AI Act).

## Struttura

```
retrieval_augmented_generation/
├── config/              # Configurazioni per domini e modelli
├── prompts/             # Template di prompt per tutti i pattern
├── schemas/             # Schemi di validazione Pydantic
├── rag/                 # Implementazioni RAG (BasicRAG, factory)
├── patterns/            # Pattern agentici (baseline, routing, collaboration, etc.)
├── evaluation/          # Metriche e pipeline di valutazione
│   ├── metrics.py       # EM, F1, BLEU, ROUGE, METEOR, BERTScore
│   ├── retrieval.py     # Valutazione retrieval
│   └── generation.py    # Valutazione generation con tutti i pattern
├── utils/               # Utilità (caching, LLM calls, JSON parsing)
└── rag_and_qa.py        # Script principale
```

## Funzionalità Implementate

### 1. Retrieval Evaluation
- ✅ Basic RAG con FAISS
- ✅ Graph-enhanced RAG
- ✅ Metriche: Precision@K, Recall@K, F1@K
- ✅ Caching dei risultati

### 2. Generation Evaluation
- ✅ **Baseline Pattern**: LLM senza RAG
- ✅ **RAG Pattern**: LLM + RAG (FAISS)
- ✅ **RAG + Graph Pattern**: LLM + RAG + Graph enhancement
- ✅ **Routing Pattern**: Classificazione automatica del dominio (GDPR/AI Act)
- ✅ **Collaboration Pattern**: Generator + Critic con iterazioni multiple
- ✅ **Self-Refinement Pattern**: Miglioramento iterativo della risposta

### 3. Metriche di Valutazione
- ✅ Exact Match (EM)
- ✅ F1 Score
- ✅ BLEU
- ✅ ROUGE (1, 2, L)
- ✅ METEOR
- ✅ BERTScore

### 4. Sistema di Caching
- ✅ Cache per retrieval (evita chiamate FAISS duplicate)
- ✅ Cache per generation (evita chiamate LLM duplicate)
- ✅ Checkpoint per riprendere valutazioni interrotte

### 5. Supporto Modelli LLM
- ✅ Llama.cpp con modelli GGUF
- ✅ Supporto per: DeepSeek, Qwen2.5, Llama 3.x, GPT-OSS
- ✅ Wrapper asincrono per esecuzione parallela

## Utilizzo

### 1. Valutazione Retrieval
```bash
python -m retrieval_augmented_generation.rag_and_qa --mode retrieval
```

### 2. Valutazione Generation
```bash
# Esegui su tutti i domini con modello singolo
python -m retrieval_augmented_generation.rag_and_qa --mode generation --models Llama-3.2-3B-Instruct

# Esegui solo su GDPR
python -m retrieval_augmented_generation.rag_and_qa --mode generation --domain gdpr --models Llama-3.2-3B-Instruct

# Esegui con multipli modelli
python -m retrieval_augmented_generation.rag_and_qa --mode generation --models Qwen2.5-3B Llama-3.2-3B-Instruct

# Riprendi da checkpoint
python -m retrieval_augmented_generation.rag_and_qa --mode generation --resume
```

### 3. Esplora Risultati
```bash
python -m retrieval_augmented_generation.rag_and_qa --mode explore
```

### 4. Esegui Tutto
```bash
python -m retrieval_augmented_generation.rag_and_qa --mode all
```

## Parametri CLI

- `--mode`: Modalità di esecuzione (`retrieval`, `generation`, `explore`, `all`)
- `--domain`: Dominio da valutare (`gdpr`, `aiact`, `both`)
- `--models`: Lista di modelli da testare (separati da spazio)
- `--resume`: Riprendi da checkpoint esistente

## Output

### Retrieval
- Cache JSONL con predizioni per ogni query
- Metriche Precision@K, Recall@K, F1@K per k=[1,5,10,15,20]

### Generation
```
data/GDPR/generator_results/
└── YYYYMMDD_HHMMSS/
    ├── results_incremental.csv      # Metriche progressive
    ├── predictions_incremental.csv  # Predizioni complete
    └── checkpoint.json              # Stato dell'esecuzione

# Dopo l'esplorazione:
data/GDPR/generator_results/
└── results_final.csv                # Metriche consolidate da tutte le esecuzioni
```

### Format Results CSV
Colonne: `model`, `pattern`, `topk`, `n_samples`, `em`, `f1`, `bleu`, `rouge1`, `rouge2`, `rougeL`, `meteor`, `bertscore_f1`, `bertscore_p`

## Differenze con Notebook

Il modulo modulare include **TUTTE** le funzionalità presenti nel notebook:
- ✅ Tutti i pattern agentici implementati
- ✅ Sistema completo di valutazione generation
- ✅ Funzione `run_full_evaluation` per grid completa
- ✅ Funzione `load_and_explore_results` per analisi risultati
- ✅ Sistema di checkpoint e resume
- ✅ Caching avanzato per performance

## Esempio di Integrazione

```python
import asyncio
from retrieval_augmented_generation import (
    Config, init_rag, run_full_evaluation, 
    load_dataset, GRAPH_FILENAMES
)
from retrieval_augmented_generation.patterns import (
    baseline_pattern, rag_pattern, collaboration_rag_pattern
)

# Configurazione
config = Config(domain="data/GDPR", selected_emb="bge", 
                graph_filename=GRAPH_FILENAMES["GDPR"])

# Inizializzazione
rag = init_rag(config, rag_type="basic")
qa_pairs = load_dataset(config.DATASET_JSONL)

# Pattern
patterns = {
    "baseline": baseline_pattern,
    "rag": rag_pattern,
    "collaboration": collaboration_rag_pattern,
}

# Valutazione
asyncio.run(run_full_evaluation(
    models=["Llama-3.2-3B-Instruct"],
    rags_dict={"data/GDPR": rag},
    domain="data/GDPR",
    qa_pairs=qa_pairs,
    model_dic=MODEL_DIC,
    patterns=patterns,
    topk_values=[1, 5, 10],
    output_dir=config.GENERATOR_RESULTS_DIR,
))
```

## Note Tecniche

### Performance
- Il caching riduce il tempo di esecuzione del ~80% su valutazioni successive
- Il sistema di checkpoint permette di interrompere e riprendere senza perdere progressi
- L'esecuzione asincrona ottimizza l'uso della GPU

### Requisiti
```bash
pip install transformers sentence-transformers faiss-cpu networkx
pip install llama-cpp-python huggingface-hub
pip install evaluate sacrebleu rouge-score bert-score nltk
pip install tqdm pandas numpy pydantic json-repair
```

### Troubleshooting

**Errore: "CUDA out of memory"**
- Riduci `limit_per_model` nella chiamata a `run_full_evaluation`
- Usa modelli più piccoli (es. Qwen2.5-3B invece di Llama-3.1-8B)

**Errore: "ModuleNotFoundError"**
- Verifica che tutte le dipendenze siano installate
- Esegui da directory root: `python -m retrieval_augmented_generation.rag_and_qa`

**Valutazione troppo lenta**
- Abilita `--resume` per riprendere da checkpoint
- Riduci il numero di pattern testati
- Usa `limit_per_model=50` per test rapidi
