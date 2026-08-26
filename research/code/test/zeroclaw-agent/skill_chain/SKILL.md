---
name: chroma-query-executor
version: 1.0
description: Runs my custom Python process on the local machine.
permissions:
  - local_execution
---

### Description
This skill triggers a local Python scripts 'chroma_email_skill.py`, `chroma_whatsapp_skill.py`, `chroma_discord_skill.py`,`chroma_query_methods.py` which processes user query.
- Install once: `~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py`
- Install once: `~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_email_skill.py`
- Install once: `~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_whatsapp_skill.py`
- Install once: `~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_discord_skill.py`

## Requirements
- Python 3.10.18
- chromadb 0.6.3

### Execute channel specific workclow, Query ChromaDB using query_str for a specific media type ("image" | "video" | "text") media type then convert in to channel specific format and send the informationg to specific email address
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_email_skill.py "media type" "email address" ['query_str',]
``` 
### Execute channel specific workclow, Query ChromaDB using query_str for a specific media type ("image" | "video" | "text") media type then convert in to channel specific format and send the informationg to specific whatsapp id
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_whatsapp_skill.py "media type" "whatsapp id" ['query_str',]
``` 
### Execute channel specific workclow, Query ChromaDB using query_str for a specific media type ("image" | "video" | "text") media type then convert in to channel specific format and send the informationg to specific discord id
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_discord_skill.py "media type" "discord id" ['query_str',]
``` 
### Usage
Use this tool whenever the user requests a chromadb query results.

### Get Multiple Collections Count
Get Collections Count for all chromadb Collections and return list with counts for all collections in chromadb:

```python
from skills.chroma-query-executor.scripts.chroma_query_methods import get_collection_count
get_collections_count()
```
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py get_collection_count
```

### Query Image Collection and return results
Query Image Collection in chromadb using query stings provided in list and return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_image_collection ['query_str',]
```
### Query Video Collection and return results
Query Video Collection in chromadb using query stings provided in list and return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_video_collection ['query_str',]
```

### Query Video Collection and return results
Query Video Collection in chromadb using query stings provided in list and return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_video_collection_uri ['query_str',]
```

### Query Text Collection and return results
Query Text Collection in chromadb using query stings provided in list and return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_text_collection ['query_str',]
```

### Query Image Collection with Metadata and return results
Query Image Collection in chromadb using query stings provided in list, src_filter and ts_filter then return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_with_image_metadata ['query_str',] src_filter ts_filter
```


### Query Video Collection with Metadata and return results
Query Video Collection in chromadb using query stings provided in list, src_filter and ts_filter then return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_with_video_metadata ['query_str',] src_filter ts_filter
```

### Query Text Collection with Metadata and return results
Query Video Collection in chromadb using query stings provided in list, src_filter and ts_filter then return python dictionary object from chromadb:
```bash
python3 ~/.openclaw/workspace/skills/chroma-query-executor/scripts/chroma_query_methods.py query_with_text_metadata ['query_str',] src_filter ts_filter
```