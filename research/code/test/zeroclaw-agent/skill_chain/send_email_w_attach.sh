#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: bmadhekar@gmail.com
To: $1
Subject: ZMedia - $(date +%m/%d/%Y)

Hello,

One years ago,

rubric - $3
narrative  - $4

<#part filename="$2" name="basename $2">
<#/part>
EOF