#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: bmadhekar@gmail.com
To: $1
Subject: "$2 - $(date +%F)"

Hello,

Please find your attached document generated dynamically.

<#part filename="$3" name="$4">
<#/part>
EOF