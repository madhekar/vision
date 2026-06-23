---
name: himalaya-repl
version: 1.0.0
description: Manage and interact with email accounts using the interactive Himalaya REPL. Allows reading, sending, searching, and organizing IMAP/SMTP mail.
allowed-tools:
  - bash
  - terminal
  - read_file
  - write_file
scripts: []
dependencies:
  - himalaya-repl
---

# Himalaya REPL Skill

## Overview
Use this skill when tasked with managing, reading, or sending emails. The `himalaya-repl` runs a continuous session which makes it faster and more state-aware than single-shot CLI commands. 

## Setup and Verification
Verify the REPL is available and configured:
1. Start the REPL: `himalaya-repl`
2. Check your connection: `account list`
3. If not configured, run: `account configure` (or `himalaya account configure` from your shell)

## Common Workflows (Inside the REPL)

### 1. Reading & Searching
```text
# Select a folder and list envelopes (summaries)
folder select INBOX
envelope list

# Search for specific emails
envelope list from alice@example.com subject invoice

# Read a specific email thread by ID
message read 42

# View full MIME source with headers
message export 42 --full
```

### 2. Composing & Replying
```text
# Start composing an email
message write

# Reply to an existing email by ID
message reply 42

# Forward an existing email
message forward 42
```
*Note: For attachments and rich formatting, use MML (Mail Markup Language).*

### 3. Organizing
```text
# Copy or move an email to a specific folder
message copy 42 Archive
message move 42 Trash

# Add or remove read/unread flags
flag add 42 --flag seen
flag remove 42 --flag seen

# Delete an email
message delete 42
```

## Anti-Patterns & Gotchas
* **IDs change dynamically:** Message IDs are relative to the currently active folder. Always re-list envelope contents using `envelope list` after navigating folders.
* **Sensitive Data:** Do not echo passwords into the chat or terminal history. Use a password manager or system keyring with Himalaya's `config.toml`.
* **Query Formatting:** When using REPL search filters, spaces must be enclosed in quotes.
  * Correct: `envelope list 'subject "Verification Code"'`
* **Safety First:** Always ask for confirmation before deleting, moving, or sending bulk messages. 

## References
* Read `references/message-composition.md` for full MML syntax and attachment handling.
* Read `references/configuration.md` for account setup and keyring authentication.
