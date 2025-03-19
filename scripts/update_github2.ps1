# Change to your project directory if needed
# cd "path/to/your/project"

# Initialize variables
$commitMessage = "Updated project files"

# Add all files (including those in subdirectories)
git add -A

# Commit changes
git commit -m $commitMessage

# Push to remote repository
git push origin main 