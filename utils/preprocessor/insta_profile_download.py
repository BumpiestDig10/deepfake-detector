import instaloader
import time
import random
import os
import json



# --- 2. State Management Functions ---
def load_state():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            print(f"🔄 Resuming from log: {LOG_NAME}")
            return json.load(f)
    
    return {"photo_idx": 1, "video_idx": 1, "downloaded_ids": []}

def save_state(state):
    with open(LOG_PATH, "w") as f:
        json.dump(state, f, indent=4)

# --- 3. Main Execution Block ---
# Configuration & Constants
TARGET_USER = input(f"Enter Instagram username to download: ")
# BURNER_USER = "andrew.bose.420" # Uncomment to use burner account
# BURNER_PASS = "abcdefghijklmnopqrstuvwxyz" # Uncomment to use burner account
LOG_DIR = "instaProfiles/index_logs"
LOG_NAME = f"{TARGET_USER}_downloadIndex.json"
LOG_PATH = os.path.join(LOG_DIR, LOG_NAME)
PHOTO_PATH = os.path.join("instaProfiles", "photos")
VIDEO_PATH = os.path.join("instaProfiles", "videos")

L = instaloader.Instaloader(
    download_geotags=False, download_comments=False, 
    save_metadata=False, download_video_thumbnails=False
)

state = load_state()

try:
    # L.login(BURNER_USER, BURNER_PASS) # Uncomment to use burner account
    profile = instaloader.Profile.from_username(L.context, TARGET_USER)
    
    print(f"📸 Starting download for {TARGET_USER}...")

    for index, post in enumerate(profile.get_posts(), 1):
        if post.shortcode in state["downloaded_ids"]:
            continue

        retries = 0
        success = False
        
        while retries < 2 and not success:
            try:
                has_photo = False
                has_video = False

                if post.typename == 'GraphSidecar':
                    sub_idx = 1
                    for node in post.get_sidecar_nodes():
                        if node.is_video:
                            L.filename_pattern = f"{TARGET_USER}_video_{state['video_idx']}_{sub_idx}"
                            L.download_post(post, target=VIDEO_PATH)
                            has_video = True
                        else:
                            L.filename_pattern = f"{TARGET_USER}_photo_{state['photo_idx']}_{sub_idx}"
                            L.download_post(post, target=PHOTO_PATH)
                            has_photo = True
                        sub_idx += 1
                else:
                    if post.is_video:
                        L.filename_pattern = f"{TARGET_USER}_video_{state['video_idx']}_1"
                        L.download_post(post, target=VIDEO_PATH)
                        has_video = True
                    else:
                        L.filename_pattern = f"{TARGET_USER}_photo_{state['photo_idx']}_1"
                        L.download_post(post, target=PHOTO_PATH)
                        has_photo = True

                # Update state
                if has_photo: state["photo_idx"] += 1
                if has_video: state["video_idx"] += 1
                state["downloaded_ids"].append(post.shortcode)
                save_state(state)
                success = True

            except (instaloader.exceptions.ConnectionException, 
                    instaloader.exceptions.QueryReturnedBadRequestException) as e:
                print(f"⚠️ Rate limit or connection error: {e}. Sleeping 15m...")
                retries += 1
                time.sleep(900)

        # Anti-detection delays
        rest = random.randint(15, 25)
        print(f"⏱️ Resting for {rest} seconds before next download.")
        time.sleep(rest)
        if index % 15 == 0:
            rest = random.randint(300, 600)
            print(f"⏸️ Batch pause: {rest // 60} minutes.")
            time.sleep(rest)

except KeyboardInterrupt:
    print("\n🛑 Manual stop detected. Progress saved.")

print("🏁 Pipeline finished.")