
# Himalaya Email CLI
 
Himalaya is a CLI email client for managing emails from the terminal via IMAP/SMTP.
 
## Tips
- Message IDs are relative to the current folder — re-list after switching folders.
- Use --output json for structured output on any envelope command.
 
## Listing & Searching
    himalaya envelope list
    himalaya envelope list --folder "Sent"
    himalaya envelope list from john@example.com subject meeting
 
## Reading
    himalaya message read 42
 
## Writing & Sending
    himalaya message write -H "To:x@example.com" -H "Subject:Hello" "body"