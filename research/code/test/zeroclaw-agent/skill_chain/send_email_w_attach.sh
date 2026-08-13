#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: bmadhekar@gmail.com
To: $1
Subject: ZMedia - $(date +%m/%d/%Y)

<#part type="text/html">
<html>
  <body>
    <p>Hello,</p>
    <p><b>Archived Media found One years ago,</b></p>
    <p><b>Rubric</b> : $3</p>
    <p><b>Narrative</b>  : $4</p>
  </body>
</html>   
<#/part> 
<#part filename="$2" name="basename $2">
<#/part>
EOF