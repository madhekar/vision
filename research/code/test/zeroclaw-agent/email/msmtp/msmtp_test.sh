#!/bin/bash

# Configuration Variables
TO="bmadhekar@gmail.com"
SUBJECT="Email with Attachment"
BOUNDARY="MULTIPART-BOUNDARY-$(date +%s)"
FILE="/path/to/file.pdf"
FILENAME=$(basename "$FILE")

# Construct Raw Email Structure
(
  echo "To: $TO"
  echo "Subject: $SUBJECT"
  echo "MIME-Version: 1.0"
  echo "Content-Type: multipart/mixed; boundary=\"$BOUNDARY\""
  echo ""
  echo "--$BOUNDARY"
  echo "Content-Type: text/plain; charset=utf-8"
  echo ""
  echo "Hello, please find your attachment below."
  echo ""
  echo "--$BOUNDARY"
  echo "Content-Type: application/octet-stream; name=\"$FILENAME\""
  echo "Content-Transfer-Encoding: base64"
  echo "Content-Disposition: attachment; filename=\"$FILENAME\""
  echo ""
  base64 "$FILE"
  echo ""
  echo "--$BOUNDARY--"
) | msmtp -t
