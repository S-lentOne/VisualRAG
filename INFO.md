# This includes descriptions for every file
*It's formatted to be Directory→File→Description*

### RAG

- **embedder.py** 
  - *loads all-MiniLM-L6-v2 once and exposes embed() for both single strings and batches. Kept separate so you can swap models in one place.*

- **knowledge_base.py--
  - *reads every .txt in documents/, chunks by word count with overlap, embeds each chunk, and upserts into ChromaDB. Re-running is safe; unchanged chunks are skipped by content hash. Adding a new class = drop a .txt file and call kb.build(). Zero code changes.*

- retriever.py
  - *two retrieval modes: retrieve() for broad semantic search across all classes, and retrieve_per_class() which guarantees at least one chunk per detected label (important for low-confidence detections that might get outranked in broad search). format_context() structures the output into labelled sections ready for the LLM prompt.*

- query_builder.py
  - *converts detection labels + scores into a rich semantic query string. Has a _SCENE_HINTS table of co-occurrence patterns (e.g. {notebook, pen, calculator} → "mathematics study session") that enrich the query. Adding new scene patterns is just appending to that list.*

### Vision

- **vision.py**
  - *this is the working "main" python script for running the Project, this includes the camera selection, yolo model(might add selection options), input selection (live video, pre-recorded video, image)*


### Models

- **tr-XX.pt**
  - *This file is a model used by vision.py, for object detection via Computer Vision*
