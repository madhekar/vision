#!/bin/bash

# Construct and pipe MML template directly into himalaya template send
cat << EOF | himalaya template send
From: $0
To: $1
Subject: $2

Hello,

Please find your attached document generated dynamically.

<#part filename="$3" name="$4">
<#/part>
EOF