#!/bin/bash

# Configuration Variables
TO="$1"
SUBJECT="ZMedia - $(date +%m/%d/%Y)"
BOUNDARY="MULTIPART-BOUNDARY-$(date +%s)"
FILE="$2"
FILENAME=$(basename "$FILE")
RUBRIC="$3"
NARATIVE="$4"

# Construct Raw Email Structure
(
  echo "To: $TO"
  echo "Subject: $SUBJECT"
  echo "MIME-Version: 1.0"
  echo "Content-Type: multipart/mixed; boundary=\"$BOUNDARY\""
  echo ""
  echo "--$BOUNDARY"
  echo "Content-Type: text/html; charset=utf-8"
  echo ""
  echo "
    <html>
     <body>
      <p>Hello,</p>
      <p><b>Archived Media found One years ago,</b></p>
      <p><b>Rubric</b> : $RUBRIC</p>
      <p><b>Narrative</b>  : $NARATIVE</p>
     </body>
   </html>"
  echo ""
  echo "--$BOUNDARY"
  echo "Content-Type: application/octet-stream; name=\"$FILENAME\""
  echo "Content-Transfer-Encoding: base64"
  echo "Content-Disposition: attachment; filename=\"$FILENAME\""
  echo ""
  base64 "$FILE"
  echo ""
  echo "--$BOUNDARY--"
) | msmtp --debug -t
