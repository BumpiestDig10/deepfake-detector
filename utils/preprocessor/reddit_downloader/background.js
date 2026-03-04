// background.js

// Function to handle the actual scraping and downloading
async function startScraper(baseUrl, target, afterToken = null, pagesLoaded = 0, imageCount = 1, videoCount = 1) {
    
    // 1. Check if the user clicked the Stop button
    const settings = await chrome.storage.local.get("isStopping");
    if (settings.isStopping) {
        // Reset the flag so the user can start a new download later
        await chrome.storage.local.set({ isStopping: false });
        chrome.runtime.sendMessage({ action: "updateStatus", text: "Stopped by user. 🛑" });
        return; 
    }

    // 2. Set a safety limit so we don't download forever (e.g., 10 pages)
    if (pagesLoaded >= 10) {
        chrome.runtime.sendMessage({ action: "updateStatus", text: "Reached page limit (10). Done! ✅" });
        return;
    }

    // 3. Update the UI and build the URL
    chrome.runtime.sendMessage({ action: "updateStatus", text: `Scraping page ${pagesLoaded + 1}... 🔍` });
    const urlWithPagination = afterToken ? `${baseUrl}?after=${afterToken}` : baseUrl;

    try {
        const response = await fetch(urlWithPagination);
        const json = await response.json();
        const posts = json.data.children;

        posts.forEach(post => {
            const data = post.data;

            // Handle Galleries 🖼️
            if (data.is_gallery && data.media_metadata) {
                data.gallery_data.items.forEach(item => {
                    const url = data.media_metadata[item.media_id].s.u.replace(/&amp;/g, '&');
                    chrome.downloads.download({
                        url: url,
                        filename: `${target}/images/${target}_image_${imageCount++}.jpg`
                    });
                });
            } 
            // Handle Videos 🎥
            else if (data.is_video && data.media && data.media.reddit_video) {
                chrome.downloads.download({
                    url: data.media.reddit_video.fallback_url,
                    filename: `${target}/videos/${target}_video_${videoCount++}.mp4`
                });
            }
            // Handle Single Images 📸
            else if (data.url && (data.url.endsWith('.jpg') || data.url.endsWith('.png'))) {
                chrome.downloads.download({
                    url: data.url,
                    filename: `${target}/images/${target}_image_${imageCount++}.jpg`
                });
            }
        });

        // 4. Check for the next page
        const nextToken = json.data.after;
        if (nextToken) {
            // Wait 1 second before fetching the next page to be polite to the API
            setTimeout(() => {
                startScraper(baseUrl, target, nextToken, pagesLoaded + 1, imageCount, videoCount);
            }, 1000);
        } else {
            chrome.runtime.sendMessage({ action: "updateStatus", text: `${target} completely scraped! 🎉` });
        }

    } catch (error) {
        console.error("Scrape Error:", error);
        chrome.runtime.sendMessage({ action: "updateStatus", text: "Error fetching data. ❌" });
    }
}

// Listener to start the process from the popup
chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "start") {
        // Ensure the stop flag is false before we begin
        chrome.storage.local.set({ isStopping: false }, () => {
            const target = message.target;
            const baseUrl = message.type === "subreddit" 
                ? `https://www.reddit.com/r/${target}.json` 
                : `https://www.reddit.com/user/${target}/submitted.json`;

            startScraper(baseUrl, target);
        });
    }
});