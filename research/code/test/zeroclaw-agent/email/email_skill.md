---
name: Himalaya Email REPL
description: Manages emails directly from the terminal using Himalaya CLI. Use for listing, reading, searching, composing, replying, forwarding, and organizing mail.
---

## 🚀 When to Use
Use this skill when asked to interact with, search, send, or manage emails. The AI should execute commands via the `himalaya-repl` interactive environment.

## 🛠️ Himalaya CLI Commands
Before running commands, orient the model by checking the current folder or listing accounts.

- `himalaya account list` : Display available email accounts.
- `himalaya folder list` : List folders (Inbox, Sent, Trash) for the currently active account.
- `himalaya folder select <folder>` : Switch to a specific folder. 
- `himalaya list` : List paginated emails in the active folder.
- `himalaya read <id>` : Read a specific email by its index/ID.
- `himalaya search <query>` : Run a natural language search or keyword filter on the mailbox.
- `himalaya write` : Compose a new email (uses interactive editor).

## ✍️ Composing & Replying
- For simple replies, use `himalaya reply <id>`.
- For forwards, use `himalaya forward <id>`.
- For rich emails, attachments, or HTML, use MML (MIME Meta Language) syntax before sending via direct pipelines.

## ⚠️ Guardrails & Gotchas
1. **Message IDs are local:** Message IDs are relative to the *currently selected folder*. Always re-list after folder changes to get the correct message ID.
2. **Never guess credentials:** If an account isn't configured, use `himalaya-repl` interactive wizard to set up credentials, or ask the user for assistance.
3. **Always Confirm:** Prior to sending emails, deleting a message, or moving folders, ask the user for confirmation.
