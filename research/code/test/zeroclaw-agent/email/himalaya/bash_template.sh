
#!/bin/bash

# Define dynamic variables
SENDER="from@example.com"
RECIPIENT="to@example.com"
SUBJECT="Report for - $(date +%F)"
FILE_PATH="/abs/path/to/report.pdf"
DISPLAY_NAME="dynamic-report-name.pdf"

# Construct and pipe MML template directly into himalaya template send
cat <<EOF | himalaya template send
From: $SENDER
To: $RECIPIENT
Subject: $SUBJECT

Hello,

Please find your attached document generated dynamically.

<#part filename="$FILE_PATH" name="$DISPLAY_NAME">
<#/part>
EOF