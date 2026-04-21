## Introduction

This repo is for a project aimed to build a computer vision and deep learning based software, which provides aid to those with bad vision/memory, mostly aimed towards (but not limited to) the elderly, and visually impared. 

The software accpets input in forms of Live Video (via an in-built webcam or an external webcam), a pre-recorded Video, or an Image, you can then question the bot with what you may want to know regarding the provided media.

For installation plese follow the steps given below:

## Installation

### Manual Installation

- **Requirements**

There are some python packages and libraries, you'll need to install, these are given in the requirements.txt, please make sure you install these according to your preference, using

```python
pip install ddgs pillow requests tqdm
```
*Other requirements should be pre-installed*

*However, if you encounter any other errors, please look in the requirements.txt and install any of the packages not yet installed.*

*to check which packages you already have installed, use:*

```pip freeze```

- **Dataset**

Please refer to Dataset/README.md for more info. 

### Script Installation 

(will be added soon)

---
## Script Information [?](action:"yet to decide")

### RAG Scripts

embedder.py — loads all-MiniLM-L6-v2 once and exposes embed() for both single strings and batches. Kept separate so you can swap models in one place.

knowledge_base.py — reads every .txt in documents/, chunks by word count with overlap, embeds each chunk, and upserts into ChromaDB. Re-running is safe — unchanged chunks are skipped by content hash. Adding a new class = drop a .txt file and call kb.build(). Zero code changes.

retriever.py — two retrieval modes: retrieve() for broad semantic search across all classes, and retrieve_per_class() which guarantees at least one chunk per detected label (important for low-confidence detections that might get outranked in broad search). format_context() structures the output into labelled sections ready for the LLM prompt.

query_builder.py — converts detection labels + scores into a rich semantic query string. Has a _SCENE_HINTS table of co-occurrence patterns (e.g. {notebook, pen, calculator} → "mathematics study session") that enrich the query. Adding new scene patterns is just appending to that list.
