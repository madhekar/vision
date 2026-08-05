#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: bmadhekar@gmail.com
To: $1
Subject: ZMedia - $(date +%m/%d/%Y)

Hello,

One years ago,

rubric - $2
narrative  - $4

<#part filename="$3" name="basename $3">
<#/part>
EOF