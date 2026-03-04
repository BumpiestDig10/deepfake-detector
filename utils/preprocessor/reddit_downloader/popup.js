// Helper to send the start message
function startDownload(type, id) {
    const val = document.getElementById(id).value.trim();
    if (val) {
        chrome.runtime.sendMessage({ action: "start", type: type, target: val });
    }
}

document.getElementById('subBtn').addEventListener('click', () => startDownload("subreddit", "subInput"));
document.getElementById('userBtn').addEventListener('click', () => startDownload("user", "userInput"));

// Stop button listener
document.getElementById('stopBtn').addEventListener('click', () => {
    chrome.storage.local.set({ isStopping: true });
    document.getElementById('statusMessage').innerText = "Stopping...";
});

// Listener for status updates from background.js
chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "updateStatus") {
        document.getElementById('statusMessage').innerText = message.text;
    }
});