import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer
from PIL import Image
import imagehash
import requests
import hashlib
import random
from collections import defaultdict

# Paths
SOURCE_DIR = "/home/pi/Pictures"
DUPLICATE_DIR = "/home/pi/Duplicate_Images"

# Ensure the Duplicate_Images directory exists
os.makedirs(DUPLICATE_DIR, exist_ok=True)

# Delay timer
DELAY_SECONDS = 15 * 60  # 15 minutes

# Timer to debounce multiple events
process_timer = None

# Telegram Bot Configuration
BOT_TOKEN = "your_bot_token"  # Replace with your bot's API token
CHAT_ID = "your_chat_id"      # Replace with your chat ID

# Functions
def send_telegram_message(file_name, original_path, destination_path):
    """Send a humorous Telegram message about duplicate detection."""
    # A list of humorous comments
    comments = [
        "Caught red-handed! This duplicate won't clutter your albums anymore.",
        "One less duplicate to worry about. Your photo library thanks you!",
        "Another sneaky duplicate bites the dust!",
        "Detective Bot strikes again! This file has been safely relocated.",
        "Oops, you uploaded this twice! No worries, I’ve got it sorted.",
        "Cleaning up your duplicates, one file at a time!"
    ]

    # Choose a random comment
    comment = random.choice(comments)

    message = (
        f"📷 Duplicate Detected!\n\n"
        f"🖼️ File: `{file_name}`\n"
        f"📂 Original Location: `{original_path}`\n"
        f"🚮 Moved to: `{destination_path}`\n\n"
        f"{comment}"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Telegram message sent!")
        else:
            print(f"Failed to send Telegram message: {response.status_code}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def get_image_resolution(image_path):
    """Get resolution (width x height) of an image."""
    with Image.open(image_path) as img:
        return img.size[0] * img.size[1]

def make_unique_filename(file_path, target_dir):
    """Generate a unique file name based on its original path."""
    base_name = os.path.basename(file_path)
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]  # Short hash of file path
    unique_name = f"{file_hash}_{base_name}"
    return os.path.join(target_dir, unique_name)

def handle_duplicate_group(duplicates):
    """Handle a group of duplicate images, keeping the best version."""
    # Sort duplicates by resolution (largest first)
    duplicates.sort(key=lambda img: get_image_resolution(img), reverse=True)

    # Keep the first (highest quality) image, move the rest
    master_image = duplicates[0]
    for duplicate in duplicates[1:]:
        # Generate a unique name and move the duplicate
        dest = make_unique_filename(duplicate, DUPLICATE_DIR)
        shutil.move(duplicate, dest)
        print(f"Moved {duplicate} to {dest}")

        # Send a Telegram notification
        file_name = os.path.basename(duplicate)
        send_telegram_message(file_name, duplicate, dest)

def find_duplicates_grouped(source_dir):
    """Find groups of duplicate images and handle them together."""
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    # Dictionary to group duplicates based on their hash
    hash_groups = defaultdict(list)

    # Generate perceptual hashes and group images by hash
    for image_path in image_paths:
        try:
            image_hash = str(imagehash.average_hash(Image.open(image_path)))
            hash_groups[image_hash].append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Handle each group of duplicates
    for hash_value, duplicates in hash_groups.items():
        if len(duplicates) > 1:  # Only process if there are duplicates
            handle_duplicate_group(duplicates)

# Watchdog event handler
class ImageFolderHandler(FileSystemEventHandler):
    """Handles events in the monitored directory."""

    def on_any_event(self, event):
        """Trigger duplicate detection after delay on any directory change."""
        global process_timer

        if process_timer:
            process_timer.cancel()

        # Start a new timer
        process_timer = Timer(DELAY_SECONDS, run_duplicate_detection)
        process_timer.start()

def run_duplicate_detection():
    """Run duplicate detection."""
    print("Starting duplicate detection using perceptual hashing...")
    find_duplicates_grouped(SOURCE_DIR)
    print("Duplicate detection completed.")

if __name__ == "__main__":
    # Start watching the directory
    print(f"Watching directory: {SOURCE_DIR}")
    event_handler = ImageFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=SOURCE_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()