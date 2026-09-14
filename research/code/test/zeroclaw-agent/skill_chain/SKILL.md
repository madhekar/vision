---
name: chroma-query-executor
version: 1.0
description: Runs my custom Python process on the local machine.
emoji: 📊
bins:
  - python3
permissions:
  - local_execution
---

# Description
This skill triggers a local Python scripts 'chroma_email_skill.py', 'chroma_whatsapp_skill.py', 'chroma_discord_skill.py', 'chroma_email_metadata_skill.py', 'chroma_whatsapp_metadata_skill.py', 'chroma_discord_metadata_skill.py'.


## Requirements
- Python 3.10.18
- chromadb 0.6.3

### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings
```bash
python3 baseDir/scripts/chroma_email_skill.py "media type" "email address" ['query_str',]
``` 

### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings
```bash
python3 baseDir/scripts/chroma_whatsapp_skill.py "media type" "whatsapp id" ['query_str',]
``` 

### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings
```bash
python3 baseDir/scripts/chroma_discord_skill.py "media type" "discord id" ['query_str',]
``` 


### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings, source name, start datetime. end datetime
```bash
python3 baseDir/scripts/chroma_email_skill.py "media type" "email address" ['query_str',] "source name" "start datetime" "end datetime"
``` 

### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings, source name, start datetime. end datetime
```bash
python3 baseDir/scripts/chroma_whatsapp_skill.py "media type" "whatsapp id" ['query_str',] "source name" "start datetime" "end datetime"
``` 

### Execute channel specific workflow method with input arguments provided shch as media type, channel specific target id, list of query strings, source name, start datetime. end datetime
```bash
python3 baseDir/scripts/chroma_discord_skill.py "media type" "discord id" ['query_str',] "source name" "start datetime" "end datetime"
```