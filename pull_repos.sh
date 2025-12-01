#!/bin/bash

# Script to pull in and combine other repositories
# Usage: ./pull_repos.sh <repo-url-1> <repo-url-2> ...

set -e

# GitHub token (you can also set this as environment variable)
GITHUB_TOKEN="${GITHUB_TOKEN:-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}"

# Directory to store pulled repos
REPOS_DIR="./pulled_repos"

# Create directory for pulled repos
mkdir -p "$REPOS_DIR"

# Function to clone or update a repo
pull_repo() {
    local repo_url=$1
    local repo_name=$(basename "$repo_url" .git)
    local repo_path="$REPOS_DIR/$repo_name"
    
    echo "Processing: $repo_url"
    
    # If repo URL doesn't include token, add it for GitHub repos
    if [[ "$repo_url" == *"github.com"* ]] && [[ "$repo_url" != *"$GITHUB_TOKEN"* ]]; then
        # Convert https://github.com/user/repo to https://token@github.com/user/repo
        repo_url=$(echo "$repo_url" | sed "s|https://github.com|https://${GITHUB_TOKEN}@github.com|")
    fi
    
    if [ -d "$repo_path" ]; then
        echo "  Updating existing repo: $repo_name"
        cd "$repo_path"
        git pull || echo "  Warning: Could not pull updates"
        cd - > /dev/null
    else
        echo "  Cloning new repo: $repo_name"
        git clone "$repo_url" "$repo_path" || {
            echo "  Error: Failed to clone $repo_url"
            return 1
        }
    fi
    
    echo "  ✓ Done: $repo_name"
    echo ""
}

# Process all provided repo URLs
if [ $# -eq 0 ]; then
    echo "Usage: $0 <repo-url-1> <repo-url-2> ..."
    echo ""
    echo "Example:"
    echo "  $0 https://github.com/user/repo1.git https://github.com/user/repo2.git"
    exit 1
fi

for repo_url in "$@"; do
    pull_repo "$repo_url"
done

echo "All repositories pulled successfully!"
echo "Repositories are in: $REPOS_DIR"
echo ""
echo "To merge a specific repo into the current project, you can:"
echo "  1. Copy files: cp -r $REPOS_DIR/repo-name/* ./"
echo "  2. Use git subtree: git subtree add --prefix=subdir $repo_url main"
echo "  3. Use git submodule: git submodule add $repo_url subdir"

