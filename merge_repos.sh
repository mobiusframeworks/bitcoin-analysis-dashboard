#!/bin/bash

# Script to merge other repositories into the current project
# Usage: ./merge_repos.sh <repo-url> <target-directory> [branch]

set -e

GITHUB_TOKEN="${GITHUB_TOKEN:-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}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <repo-url> <target-directory> [branch]"
    echo ""
    echo "Examples:"
    echo "  $0 https://github.com/user/repo.git ./integrated-repo"
    echo "  $0 https://github.com/user/repo.git ./integrated-repo main"
    exit 1
fi

REPO_URL=$1
TARGET_DIR=$2
BRANCH=${3:-main}

# Add token to GitHub URLs
if [[ "$REPO_URL" == *"github.com"* ]] && [[ "$REPO_URL" != *"$GITHUB_TOKEN"* ]]; then
    REPO_URL=$(echo "$REPO_URL" | sed "s|https://github.com|https://${GITHUB_TOKEN}@github.com|")
fi

echo "Merging repository into: $TARGET_DIR"
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

# Method 1: Using git subtree (preserves history)
echo "Method 1: Using git subtree (recommended - preserves history)"
read -p "Use git subtree? (y/n): " use_subtree

if [ "$use_subtree" = "y" ] || [ "$use_subtree" = "Y" ]; then
    echo "Adding as git subtree..."
    git subtree add --prefix="$TARGET_DIR" "$REPO_URL" "$BRANCH" --squash
    echo "✓ Repository merged using git subtree"
    exit 0
fi

# Method 2: Clone and copy files (simpler, no history)
echo "Method 2: Clone and copy files (no history preserved)"
TEMP_DIR=$(mktemp -d)
echo "Cloning to temporary directory..."

git clone -b "$BRANCH" "$REPO_URL" "$TEMP_DIR" || {
    echo "Error: Failed to clone repository"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy files (excluding .git)
echo "Copying files to $TARGET_DIR..."
rsync -av --exclude='.git' "$TEMP_DIR/" "$TARGET_DIR/" || cp -r "$TEMP_DIR"/* "$TARGET_DIR/"

# Clean up
rm -rf "$TEMP_DIR"

echo "✓ Files copied to $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the files in $TARGET_DIR"
echo "  2. git add $TARGET_DIR"
echo "  3. git commit -m 'Merge repository: $REPO_URL'"

