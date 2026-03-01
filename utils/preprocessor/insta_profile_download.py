import instaloader
import time
import random
import os
import json
import centralLogging as cl



# --- 2. State Management Functions ---
def load_state():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            logger.info(f"Resuming from log: {LOG_NAME}")
            return json.load(f)
    
    return {f"{TARGET_USER}_photo_idx": 1, f"{TARGET_USER}_video_idx": 1, "downloaded_ids": []}

def save_state(state):
    with open(LOG_PATH, "w") as f:
        json.dump(state, f, indent=4)

# --- 3. Main Execution Block ---

logger = cl.get_logger(console_level="INFO", file_level="DEBUG")

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
    
    logger.info(f"Starting download for {TARGET_USER}...")

    for index, post in enumerate(profile.get_posts(), 1):
        if post.shortcode in state["downloaded_ids"]:
            logger.info(f"Skipping already downloaded post: {post.shortcode}")
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
                            L.filename_pattern = f"{TARGET_USER}_photo_{state[f'{TARGET_USER}_photo_idx']}_{sub_idx}"
                            L.download_post(post, target=PHOTO_PATH)
                            has_photo = True
                        sub_idx += 1
                else:
                    if post.is_video:
                        L.filename_pattern = f"{TARGET_USER}_video_{state[f'{TARGET_USER}_video_idx']}_1"
                        L.download_post(post, target=VIDEO_PATH)
                        has_video = True
                    else:
                        L.filename_pattern = f"{TARGET_USER}_photo_{state[f'{TARGET_USER}_photo_idx']}_1"
                        L.download_post(post, target=PHOTO_PATH)
                        has_photo = True

                # Update state
                if has_photo: state[f"{TARGET_USER}_photo_idx"] += 1
                if has_video: state[f"{TARGET_USER}_video_idx"] += 1
                state["downloaded_ids"].append(post.shortcode)
                save_state(state)
                success = True
                logger.info(f"Downloaded post: {post.shortcode} (Photo: {has_photo}, Video: {has_video})")

            except (instaloader.exceptions.ConnectionException, 
                    instaloader.exceptions.QueryReturnedBadRequestException) as e:
                logger.warning(f"Rate limit or connection error: {e}. Sleeping 15m...")
                retries += 1
                time.sleep(900)

        # Anti-detection delays
        if index % 15 == 0:
            rest = random.randint(180, 420)
            logger.debug(f"Batch pause: {rest // 60} minutes.")
            time.sleep(rest)
        else:
            rest = random.randint(5, 25)
            logger.debug(f"Resting for {rest} seconds before next download.")
            time.sleep(rest)

except KeyboardInterrupt:
    logger.warning("\nManual stop detected. Progress saved.")

logger.info("Pipeline finished.")