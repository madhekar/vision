---
name: chroma_query_executor
version: 1.0
description: Runs my custom Python class process on the local machine.
permissions:
  - local_execution
---
~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py
### Description
This skill triggers a local Python script `chroma_query.py` which processes user query.
- Install once: `~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py`

## Requirements
- Python 3.10.18
- chromadb 0.6.3


### Usage
Use this tool whenever the user requests a chromadb query execution.

### Get Multiple Collections Count
Get Collection Counts for all chromadb Collections and return list with counts for all collections in chromadb:

```python
from skills.chroma_query_executor.scripts.chroma_query import getCollectionCount
getCollectionCount()
```
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py getCollectionCount
```

### Query Image Collection and return results
Query Image Collection in chromadb using query stings provided in list and return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_image_collection ['query_str',]
```
### Query Video Collection and return results
Query Video Collection in chromadb using query stings provided in list and return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_video_collection ['query_str',]
```

### Query Text Collection and return results
Query Text Collection in chromadb using query stings provided in list and return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_text_collection ['query_str',]
```

### Query Image Collection with Metadata and return results
Query Image Collection in chromadb using query stings provided in list, src_filter and ts_filter then return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_with_image_metadata ['query_str',] src_filter ts_filter
```


### Query Video Collection with Metadata and return results
Query Video Collection in chromadb using query stings provided in list, src_filter and ts_filter then return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_with_video_metadata ['query_str',] src_filter ts_filter
```

### Query Text Collection with Metadata and return results
Query Video Collection in chromadb using query stings provided in list, src_filter and ts_filter then return list of results from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma_query_executor/scripts/chroma_query.py query_with_text_metadata ['query_str',] src_filter ts_filter
```



