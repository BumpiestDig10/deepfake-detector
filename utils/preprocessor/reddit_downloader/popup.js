// ─── Archive controls ────────────────────────────────────────────────────────

function startArchive(type, inputId) {
    const target = document.getElementById(inputId).value.trim();
    if (target) {
        chrome.runtime.sendMessage({ action: "start", type, target });
    }
}

document.getElementById("subBtn").addEventListener("click", () => startArchive("subreddit", "subInput"));
document.getElementById("userBtn").addEventListener("click", () => startArchive("user", "userInput"));

document.getElementById("stopBtn").addEventListener("click", () => {
    chrome.storage.local.set({ isStopping: true });
    document.getElementById("statusMessage").innerText = "Stopping...";
});

chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "updateStatus") {
        document.getElementById("statusMessage").innerText = message.text;
    }
});

// ─── Backup helpers ──────────────────────────────────────────────────────────

function setBackupStatus(text, isError = false) {
    const el = document.getElementById("backupStatus");
    el.innerText = text;
    el.style.color = isError ? "#c62828" : "#2e7d32";
    // Clear message after 4 seconds
    setTimeout(() => { el.innerText = ""; }, 4000);
}

// ─── Export ──────────────────────────────────────────────────────────────────

document.getElementById("exportBtn").addEventListener("click", async () => {
    const data = await chrome.storage.local.get("history");
    const history = data.history || {};

    if (Object.keys(history).length === 0) {
        setBackupStatus("Nothing to export yet.", true);
        return;
    }

    // Build a timestamped filename: reddit_archive_2024-06-01.json
    const date = new Date().toISOString().slice(0, 10);
    const filename = `reddit_archive_${date}.json`;

    const blob = new Blob([JSON.stringify(history, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href     = url;
    a.download = filename;
    a.click();

    // Clean up the object URL after a short delay
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setBackupStatus(`Exported as ${filename}`);
});

// ─── Import ──────────────────────────────────────────────────────────────────

document.getElementById("importBtn").addEventListener("click", () => {
    // Trigger the hidden file picker
    document.getElementById("importFile").click();
});

document.getElementById("importFile").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset so the same file can be re-imported if needed
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = async (event) => {
        let incoming;
        try {
            incoming = JSON.parse(event.target.result);
        } catch {
            setBackupStatus("Invalid JSON file.", true);
            return;
        }

        // Validate: must be a plain object whose values are also objects (not arrays/null)
        const isValid = typeof incoming === "object" &&
            incoming !== null &&
            !Array.isArray(incoming) &&
            Object.values(incoming).every(v => v !== null && typeof v === "object" && !Array.isArray(v));

        if (!isValid) {
            setBackupStatus("File doesn't look like a valid archive backup.", true);
            return;
        }

        // Migrate entries from the old key format (newest/oldest/reached_bottom)
        // to the new format (newest_id/oldest_id/profile_completed) if needed
        for (const [target, entry] of Object.entries(incoming)) {
            if (!("newest_id" in entry)) {
                incoming[target] = {
                    newest_id:         entry.newest        ?? null,
                    oldest_id:         entry.oldest        ?? null,
                    profile_completed: entry.reached_bottom ?? false
                };
            }
        }

        // Merge with existing history (imported values win on conflict)
        const existing = await chrome.storage.local.get("history");
        const merged = Object.assign({}, existing.history || {}, incoming);
        await chrome.storage.local.set({ history: merged });

        const count = Object.keys(incoming).length;
        setBackupStatus(`Imported ${count} target${count !== 1 ? "s" : ""}`);
    };

    reader.readAsText(file);
});