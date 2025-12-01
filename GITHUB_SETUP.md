# How to Upload to GitHub - Simple Steps

## Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in (or create an account)
2. Click the **"+"** button in the top right → **"New repository"**
3. Name it (e.g., "ds4200-dashboard" or "stock-market-dashboard")
4. Make it **Public** (so GitHub Pages will work)
5. **DO NOT** check "Initialize with README" (we already have files)
6. Click **"Create repository"**

## Step 2: Connect and Upload

After creating the repository, GitHub will show you commands. But I'll run them for you!

**Just tell me:**
- Your GitHub username
- The repository name you created

Or you can run these commands yourself (replace YOUR_USERNAME and REPO_NAME):

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 3: Enable GitHub Pages (Make it Live!)

1. Go to your repository on GitHub
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar)
4. Under "Source", select **main** branch
5. Click **Save**
6. Wait 1-2 minutes, then visit: `https://YOUR_USERNAME.github.io/REPO_NAME/`

That's it! Your website will be live! 🎉

