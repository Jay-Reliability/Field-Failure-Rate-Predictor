# deploy.ps1
# Helper script to initialize and push repository to GitHub for Streamlit Cloud deployment

# 1. Ask user for GitHub repository URL
$repoUrl = Read-Host -Prompt "Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git)"

if ([string]::IsNullOrEmpty($repoUrl)) {
    Write-Host "Error: Repository URL cannot be empty." -ForegroundColor Red
    Exit
}

# 2. Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Error: Git is not installed on this system. Please install Git first." -ForegroundColor Red
    Exit
}

# 3. Initialize git, commit and push
Write-Host "`nInitializing Git Repository..." -ForegroundColor Cyan
git init

Write-Host "Adding files..." -ForegroundColor Cyan
git add .

Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initialize Failure Rate Prediction System for Streamlit Cloud deployment"

Write-Host "Configuring main branch..." -ForegroundColor Cyan
git branch -M main

# Check if origin already exists and update/add it
$remoteExists = git remote | Select-String "^origin$"
if ($remoteExists) {
    git remote set-url origin $repoUrl
    Write-Host "Updated existing Git remote origin to: $repoUrl" -ForegroundColor Green
} else {
    git remote add origin $repoUrl
    Write-Host "Added Git remote origin: $repoUrl" -ForegroundColor Green
}

Write-Host "Pushing code to GitHub (main branch)..." -ForegroundColor Cyan
git push -u origin main --force

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "Success! Your files have been uploaded to GitHub." -ForegroundColor Green
Write-Host "Now go to https://share.streamlit.io/ to deploy your app!" -ForegroundColor Green
Write-Host "Make sure to select 'streamlit_app.py' as the Main file path." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
