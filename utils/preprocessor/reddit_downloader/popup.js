function startArchive(type, inputId) {
    const target = document.getElementById(inputId).value.trim();
    if (target) {
        chrome.runtime.sendMessage({ action: "start", type: type, target: target });
    }
}

document.getElementById('subBtn').addEventListener('click', () => startArchive("subreddit", "subInput"));
document.getElementById('userBtn').addEventListener('click', () => startArchive("user", "userInput"));

document.getElementById('stopBtn').addEventListener('click', () => {
    chrome.storage.local.set({ isStopping: true });
    document.getElementById('statusMessage').innerText = "Stopping...";
});

chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "updateStatus") {
        document.getElementById('statusMessage').innerText = message.text;
    }
});