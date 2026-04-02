// --- Logger ------------------------------------------------------------------
// View logs: chrome://extensions -> "Inspect service worker" -> Console tab
// Logs are session-only and cleared when the service worker restarts.

const LOG_PREFIX = "[RedditScraper]";

const log = {
    _style: {
        tag:   "font-weight:bold; color:#fff; padding:1px 4px; border-radius:3px;",
        reset: "font-weight:normal; color:inherit;",
        dim:   "color:#999;",
        url:   "color:#64b5f6; text-decoration:underline;",
    },

    _tag(label, color) {
        return [`%c${label}%c`, `${this._style.tag} background:${color};`, this._style.reset];
    },

    session(event, target, type) {
        const [tag, s1, s2] = this._tag(` SESSION `, "#546e7a");
        console.log(`${LOG_PREFIX} ${tag} ${event} - ${type}: %c${target}`, s1, s2, "font-weight:bold;");
    },

    stop(reason) {
        const [tag, s1, s2] = this._tag(` STOP `, "#b71c1c");
        console.log(`${LOG_PREFIX} ${tag} ${reason}`, s1, s2);
    },

    fetch(mode, pageNum, postCount, afterToken) {
        const color = mode === "CATCH-UP" ? "#e65100" : "#1565c0";
        const [tag, s1, s2] = this._tag(` ${mode} `, color);
        console.log(
            `${LOG_PREFIX} ${tag} Page %c${pageNum}%c -> ${postCount} post(s)   after=${afterToken ?? "START"}`,
            s1, s2, "font-weight:bold;", this._style.dim
        );
    },

    idCheck(latestId, newestId, matched) {
        const tag    = matched ? "[MATCH]" : "[NEW POSTS]";
        const result = matched ? "no new posts" : "new posts detected";
        console.groupCollapsed(`${LOG_PREFIX} [ID-CHECK] ${tag} ${result}`);
        console.log(`  Latest on feed : %c${latestId}`,                        "font-weight:bold;");
        console.log(`  Stored newest  : %c${newestId ?? "(none - first run)"}`, "font-weight:bold;");
        console.groupEnd();
    },

    oldestCheck(oldestId, profileCompleted) {
        const action = profileCompleted ? "Skip backfill (already done)" : "Resume backfill from oldest ID";
        console.groupCollapsed(`${LOG_PREFIX} [OLDEST-CHECK] profile_completed=${profileCompleted}`);
        console.log(`  Stored oldest     : %c${oldestId ?? "(none)"}`, "font-weight:bold;");
        console.log(`  Profile completed : %c${profileCompleted}`,      "font-weight:bold;");
        console.log(`  Action            : ${action}`);
        console.groupEnd();
    },

    download(type, filename, url) {
        const color = type === "video" ? "#6a1b9a" : "#2e7d32";
        const [tag, s1, s2] = this._tag(` ${type.toUpperCase()} `, color);
        console.groupCollapsed(`${LOG_PREFIX} ${tag} ${filename}`, s1, s2);
        console.log(`  %cURL%c  ${url}`, "font-weight:bold;", this._style.url);
        console.groupEnd();
    },

    oldestSaved(oldestId, trigger) {
        console.log(
            `${LOG_PREFIX} [SAVED] oldest_id  [${trigger}]  -> %c${oldestId}`,
            "font-weight:bold; color:#ffb300;"
        );
    },

    boundary(postId) {
        console.log(`${LOG_PREFIX} [BOUNDARY] Catch-up boundary reached at post: %c${postId}`, "font-weight:bold;");
    },

    complete(target) {
        const [tag, s1, s2] = this._tag(` COMPLETE `, "#2e7d32");
        console.log(`${LOG_PREFIX} ${tag} All posts downloaded for: ${target}`, s1, s2);
    },

    error(context, err) {
        const [tag, s1, s2] = this._tag(` ERROR `, "#c62828");
        console.group(`${LOG_PREFIX} ${tag} ${context}`, s1, s2);
        console.error(err);
        console.groupEnd();
    },

    warn(msg) {
        console.warn(`${LOG_PREFIX} [WARN] ${msg}`);
    },
};

// --- Helpers -----------------------------------------------------------------

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
            const filename = `${target}/images/${target}_image_${item.media_id}.${ext}`;
            chrome.downloads.download({ url, filename });
            log.download("image", filename, url);
        }
        return;
    }

    // Reddit-hosted videos
    if (data.is_video && data.media?.reddit_video) {
        const vid = data.media.reddit_video;
        const filename = `${target}/videos/${target}_video_${counters.video++}.mp4`;
        chrome.downloads.download({ url: vid.fallback_url, filename });
        log.download("video", filename, vid.fallback_url);
        return;
    }

    // Direct image links
    if (data.url) {
        const lower = data.url.toLowerCase().split("?")[0];
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") ||
            lower.endsWith(".png") || lower.endsWith(".gif") ||
            lower.endsWith(".webp")) {
            const ext = lower.split(".").pop();
            const filename = `${target}/images/${target}_image_${data.id}.${ext}`;
            chrome.downloads.download({ url: data.url, filename });
            log.download("image", filename, data.url);
        }
    }
}

// --- Catch-up: download posts newer than newest_id ---------------------------

async function runCatchUp(baseUrl, target, stopAtId, counters) {
    let afterToken = null;
    let pageNum    = 0;
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
            log.error(`Catch-up page ${pageNum} - fetch failed`, e);
            sendStatus("Network error during catch-up. Check console.");
            return { stopped: false, foundBoundary };
        }

        const posts = json.data?.children ?? [];
        log.fetch("CATCH-UP", pageNum, posts.length, afterToken);

        for (const post of posts) {
            const data = post.data;
            if (data.name === stopAtId) {
                log.boundary(stopAtId);
                foundBoundary = true;
                return { stopped: false, foundBoundary };
            }
            downloadMedia(data, target, counters);
        }

        afterToken = json.data?.after ?? null;
        if (!afterToken) {
            log.warn("Catch-up ran out of pages without hitting the stored newest_id boundary.");
            break;
        }

        await delay(1000);
    }

    return { stopped: false, foundBoundary };
}

// --- Backfill: download posts older than oldest_id ---------------------------

async function runBackfill(baseUrl, target, counters) {
    const history = await getHistory(target);
    const th = history[target];

    log.oldestCheck(th.oldest_id, th.profile_completed);

    let afterToken = th.oldest_id;
    let pageNum    = 0;

    while (true) {
        if (await isStopping()) {
            await saveHistory(history);
            log.oldestSaved(th.oldest_id, "user stop");
            log.stop("Stopped by user");
            await chrome.storage.local.set({ isStopping: false });
            sendStatus("Stopped by user.");
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
            log.error(`Backfill page ${pageNum} - fetch failed`, e);
            sendStatus("Network error during backfill. Check console.");
            await saveHistory(history);
            log.oldestSaved(th.oldest_id, "error");
            return;
        }

        const posts = json.data?.children ?? [];
        log.fetch("BACKFILL", pageNum, posts.length, afterToken);

        if (posts.length === 0) {
            th.profile_completed = true;
            await saveHistory(history);
            log.complete(target);
            sendStatus("Archive complete! All posts downloaded.");
            return;
        }

        for (const post of posts) {
            downloadMedia(post.data, target, counters);
        }

        // Persist oldest_id after every page
        const lastPost = posts[posts.length - 1];
        th.oldest_id = lastPost.data.name;
        await saveHistory(history);
        log.oldestSaved(th.oldest_id, "page complete");

        afterToken = json.data?.after ?? null;
        if (!afterToken) {
            th.profile_completed = true;
            await saveHistory(history);
            log.complete(target);
            sendStatus("Archive complete! All posts downloaded.");
            return;
        }

        await delay(1000);
    }
}

// --- Main entry point --------------------------------------------------------

async function startScraper(baseUrl, target, type) {
    const counters = { image: 1, video: 1 };

    log.session("START", target, type);
    sendStatus("Checking for new posts...");

    let firstPageJson;
    try {
        const res = await fetch(baseUrl);
        firstPageJson = await res.json();
    } catch (e) {
        log.error("Failed to fetch first page", e);
        sendStatus("Failed to reach Reddit. Check your connection.");
        return;
    }

    const firstPagePosts = firstPageJson.data?.children ?? [];
    if (firstPagePosts.length === 0) {
        log.warn(`No posts found for target: ${target}`);
        sendStatus("No posts found for this target.");
        return;
    }

    const latestPostId = firstPagePosts[0].data.name;
    const history      = await getHistory(target);
    const th           = history[target];

    // Always log the ID comparison before branching
    log.idCheck(latestPostId, th.newest_id, th.newest_id === latestPostId);

    // -- FIRST RUN --
    if (!th.newest_id) {
        sendStatus("First run - starting full archive...");
        th.newest_id = latestPostId;

        for (const post of firstPagePosts) {
            downloadMedia(post.data, target, counters);
        }

        const lastOnPage = firstPagePosts[firstPagePosts.length - 1];
        th.oldest_id = lastOnPage.data.name;
        await saveHistory(history);
        log.oldestSaved(th.oldest_id, "first page");

        await runBackfill(baseUrl, target, counters);
        return;
    }

    // -- NEW POSTS EXIST --
    if (latestPostId !== th.newest_id) {
        sendStatus("New posts detected - catching up...");
        const previousNewest = th.newest_id;

        th.newest_id = latestPostId;
        await saveHistory(history);

        const { stopped } = await runCatchUp(baseUrl, target, previousNewest, counters);
        if (stopped) {
            log.stop("Stopped by user during catch-up");
            await chrome.storage.local.set({ isStopping: false });
            sendStatus("Stopped by user.");
            return;
        }

        if (!th.profile_completed) {
            sendStatus("Catch-up done. Resuming backfill of older posts...");
            await delay(500);
            await runBackfill(baseUrl, target, counters);
        } else {
            sendStatus("Catch-up complete! Archive was already fully backfilled.");
        }
        return;
    }

    // -- NO NEW POSTS --
    if (th.profile_completed) {
        log.warn(`Target fully archived, nothing to do: ${target}`);
        sendStatus("Everything is up to date!");
        return;
    }

    sendStatus("No new posts. Resuming backfill of older posts...");
    await delay(500);
    await runBackfill(baseUrl, target, counters);
}

// --- Utility -----------------------------------------------------------------

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Message listener --------------------------------------------------------

chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "start") {
        const baseUrl = message.type === "subreddit"
            ? `https://www.reddit.com/r/${message.target}/new.json?limit=100`
            : `https://www.reddit.com/user/${message.target}/submitted.json?sort=new&limit=100`;
        startScraper(baseUrl, message.target, message.type);
    }
});