// ─── Helpers ────────────────────────────────────────────────────────────────

function sendStatus(text) {
    chrome.runtime.sendMessage({ action: "updateStatus", text });
}

async function isStopping() {
    const s = await chrome.storage.local.get("isStopping");
    return !!s.isStopping;
}

async function getHistory(target) {
    const s = await chrome.storage.local.get("history");
    const history = s.history || {};
    if (!history[target]) {
        history[target] = { newest_id: null, oldest_id: null, profile_completed: false };
    }
    return history;
}

async function saveHistory(history) {
    await chrome.storage.local.set({ history });
}

function downloadMedia(data, target, counters) {
    // Gallery posts
    if (data.is_gallery && data.media_metadata && data.gallery_data) {
        for (const item of data.gallery_data.items) {
            const meta = data.media_metadata[item.media_id];
            if (!meta || !meta.s) continue;
            const url = (meta.s.u || meta.s.gif || "").replace(/&amp;/g, "&");
            if (!url) continue;
            const ext = url.includes(".gif") ? "gif" : "jpg";
            chrome.downloads.download({
                url,
                filename: `${target}/images/${target}_image_${item.media_id}.${ext}`
            });
        }
        return;
    }

    // Reddit-hosted videos
    if (data.is_video && data.media?.reddit_video) {
        const vid = data.media.reddit_video;
        chrome.downloads.download({
            url: vid.fallback_url,
            filename: `${target}/videos/${target}_video_${counters.video++}.mp4`
        });
        return;
    }

    // Direct image links
    if (data.url) {
        const lower = data.url.toLowerCase().split("?")[0];
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") ||
            lower.endsWith(".png") || lower.endsWith(".gif") ||
            lower.endsWith(".webp")) {
            const ext = lower.split(".").pop();
            chrome.downloads.download({
                url: data.url,
                filename: `${target}/images/${target}_image_${data.id}.${ext}`
            });
        }
    }
}

// ─── Catch-up: download posts newer than newest_id ──────────────────────────
// Returns the `after` token at which we stopped (null = ran out of pages),
// and whether we found the boundary post.

async function runCatchUp(baseUrl, target, stopAtId, counters) {
    let afterToken = null;
    let pageNum = 0;
    let foundBoundary = false;

    while (true) {
        if (await isStopping()) return { stopped: true, foundBoundary };

        pageNum++;
        sendStatus(`CATCH-UP: Page ${pageNum}`);

        const url = afterToken ? `${baseUrl}&after=${afterToken}` : baseUrl;
        let json;
        try {
            const res = await fetch(url);
            json = await res.json();
        } catch (e) {
            sendStatus("Network error during catch-up. Check console.");
            console.error(e);
            return { stopped: false, foundBoundary };
        }

        const posts = json.data?.children ?? [];

        for (const post of posts) {
            const data = post.data;

            // Hit the previously-known newest post — stop catch-up here
            if (data.name === stopAtId) {
                foundBoundary = true;
                return { stopped: false, foundBoundary };
            }

            downloadMedia(data, target, counters);
        }

        afterToken = json.data?.after ?? null;
        if (!afterToken) break; // Ran off the end without finding the boundary

        await delay(1000);
    }

    return { stopped: false, foundBoundary };
}

// ─── Backfill: download posts older than oldest_id ──────────────────────────

async function runBackfill(baseUrl, target, counters) {
    const history = await getHistory(target);
    const targetHistory = history[target];

    // Start pagination after the oldest post we already have
    let afterToken = targetHistory.oldest_id;
    let pageNum = 0;

    while (true) {
        if (await isStopping()) {
            // Save whatever oldest_id we've reached before stopping
            await saveHistory(history);
            await chrome.storage.local.set({ isStopping: false });
            sendStatus("Stopped by user. 🛑");
            return;
        }

        pageNum++;
        sendStatus(`BACKFILL: Page ${pageNum}`);

        const url = afterToken ? `${baseUrl}&after=${afterToken}` : baseUrl;
        let json;
        try {
            const res = await fetch(url);
            json = await res.json();
        } catch (e) {
            sendStatus("Network error during backfill. Check console.");
            console.error(e);
            await saveHistory(history);  // Save progress on error too
            return;
        }

        const posts = json.data?.children ?? [];
        if (posts.length === 0) {
            // No more posts — we've reached the very bottom
            targetHistory.profile_completed = true;
            await saveHistory(history);
            sendStatus("Archive complete! All posts downloaded. 🎉");
            return;
        }

        for (const post of posts) {
            downloadMedia(post.data, target, counters);
        }

        // Update oldest_id to the last post on this page (save after every page)
        const lastPost = posts[posts.length - 1];
        targetHistory.oldest_id = lastPost.data.name;
        await saveHistory(history);

        afterToken = json.data?.after ?? null;
        if (!afterToken) {
            // Reddit gave us posts but no next-page token → bottom reached
            targetHistory.profile_completed = true;
            await saveHistory(history);
            sendStatus("Archive complete! All posts downloaded. 🎉");
            return;
        }

        await delay(1000);
    }
}

// ─── Main entry point ────────────────────────────────────────────────────────

async function startScraper(baseUrl, target) {
    const counters = { image: 1, video: 1 };

    // ── Step 1: Fetch the very first page to learn the latest post ID ──
    sendStatus("Checking for new posts…");
    let firstPageJson;
    try {
        const res = await fetch(baseUrl);
        firstPageJson = await res.json();
    } catch (e) {
        sendStatus("Failed to reach Reddit. Check your connection.");
        console.error(e);
        return;
    }

    const firstPagePosts = firstPageJson.data?.children ?? [];
    if (firstPagePosts.length === 0) {
        sendStatus("No posts found for this target.");
        return;
    }

    const latestPostId = firstPagePosts[0].data.name;

    // ── Step 2: Load memory ──
    const history = await getHistory(target);
    const th = history[target]; // shorthand

    // ── Step 3: Decide what to do ──────────────────────────────────────

    // FIRST RUN — no history at all
    if (!th.newest_id) {
        sendStatus("First run — starting full archive…");
        th.newest_id = latestPostId;

        // Treat first page posts as the start of backfill
        // (no catch-up needed; everything is "old" by definition)
        for (const post of firstPagePosts) {
            downloadMedia(post.data, target, counters);
        }

        // oldest_id after first page = last post on that page
        const lastOnPage = firstPagePosts[firstPagePosts.length - 1];
        th.oldest_id = lastOnPage.data.name;
        await saveHistory(history);

        // Continue paging down from here
        await runBackfill(baseUrl, target, counters);
        return;
    }

    // NEW POSTS EXIST — catch up, then resume backfill
    if (latestPostId !== th.newest_id) {
        sendStatus("New posts detected — catching up…");
        const previousNewest = th.newest_id;

        // Update newest_id to the actual latest post right away
        th.newest_id = latestPostId;
        await saveHistory(history);

        const { stopped } = await runCatchUp(baseUrl, target, previousNewest, counters);
        if (stopped) {
            await chrome.storage.local.set({ isStopping: false });
            sendStatus("Stopped by user. 🛑");
            return;
        }

        // After catch-up, fall through to backfill if not yet complete
        if (!th.profile_completed) {
            sendStatus("Catch-up done. Resuming backfill of older posts…");
            await delay(500);
            await runBackfill(baseUrl, target, counters);
        } else {
            sendStatus("Catch-up complete! Archive was already fully backfilled. ✅");
        }
        return;
    }

    // NO NEW POSTS — check whether backfill is complete
    if (latestPostId === th.newest_id) {
        if (th.profile_completed) {
            sendStatus("Everything is up to date! ✅");
            return;
        }

        // Resume backfill from where we left off
        sendStatus("No new posts. Resuming backfill of older posts…");
        await delay(500);
        await runBackfill(baseUrl, target, counters);
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Message listener ────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "start") {
        const baseUrl = message.type === "subreddit"
            ? `https://www.reddit.com/r/${message.target}/new.json?limit=100`
            : `https://www.reddit.com/user/${message.target}/submitted.json?sort=new&limit=100`;
        startScraper(baseUrl, message.target);
    }
});