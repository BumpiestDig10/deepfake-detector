# Reddit Media Scraper

Reddit Media Scraper is a powerful Chrome extension designed to automatically scrape and download media—including images, videos, and GIFs—from specific subreddits or user profiles. It functions as a smart, resumable archiver that organizes downloaded files and remembers its progress so you never have to worry about downloading duplicate files.

---

## Features

*   **Comprehensive Media Support:** Automatically detects and downloads Reddit gallery posts, Reddit-hosted videos, and direct image links (supporting `.jpg`, `.jpeg`, `.png`, `.gif`, and `.webp` formats).
*   **Organized Storage:** Leverages Chrome's download manager to automatically save media into structured folders based on the target name and media type (e.g., `Downloads/[target]/images/` or `Downloads/[target]/videos/`).
*   **Smart Archiving Logic:** 
    *   **First Run:** Starts at the newest post and begins a "Backfill," paging backward through Reddit history to build your archive.
    *   **Catch-Up:** If you run the scraper again later, it will download only the newest posts until it hits the most recent post it remembers (`newest_id`), then it stops or resumes the backfill.
    *   **Resumable:** Tracks your progress using an `oldest_id` marker. If you stop the process, it can safely resume exactly where it left off.
*   **Data Portability:** Export your archive history as a timestamped JSON file (e.g., `reddit_archive_YYYY-MM-DD.json`) and import it on another device to seamlessly migrate your backup progress.

## Installation *(Standard Chrome Procedure)*

1. Download or clone the extension files into a folder on your computer.
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Enable **"Developer mode"** using the toggle switch in the top right corner.
4. Click the **"Load unpacked"** button in the top left.
5. Select the folder containing the extension's `manifest.json` file.

## How to Use

### 1. Start an Archive
1. Click the Reddit Media Scraper extension icon in your Chrome toolbar to open the popup interface.
2. Enter the exact name of the subreddit or the Reddit username you want to archive.
3. Click **"Archive Subreddit"** or **"Archive User"**. 
4. The extension will query the Reddit API in batches of 100 posts and begin downloading media automatically. The popup will display status messages like *"Checking for new posts…"* or *"BACKFILL: Page 1"*.

### 2. Pause or Stop
If you need to halt the scraping process, simply open the extension and click the **"Stop Process"** button. The extension will finish its current task, save your exact place in the Reddit feed, and display a "Stopped by user" message.

### 3. Backup and Restore History
To prevent the extension from losing its memory of what has already been downloaded (for example, if you reinstall your browser):
*   **Export:** Click the **"Export"** button under the "Backup History" section. This will download a JSON file containing your current progress for all scraped targets.
*   **Import:** Click the **"Import"** button and select a previously exported JSON file. The extension will validate the file, automatically update any older formats, and merge it with your current history.

---

## Permissions
This extension requires the following permissions to function:
*   `downloads`: To automatically save media files to your local drive.
*   `storage`: To save your scraping history (`newest_id`, `oldest_id`, `profile_completed`) locally in your browser.
*   `host_permissions`: Access to `https://www.reddit.com/*` to fetch JSON data from subreddits and user profiles.