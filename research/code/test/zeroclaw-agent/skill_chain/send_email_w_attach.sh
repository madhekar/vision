#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: bmadhekar@gmail.com
To: $1
Subject: ZMedia - $(date +%m/%d/%Y)

Hello,

<#part type=text/html>
<p><b>One years ago</b></p>,

<p><u>rubric</u></p> - $3
<p><u>narrative</u></p>  - $4
<#/part>

<#part filename="$2" name="basename $2">
<#/part>
EOF