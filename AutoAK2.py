import signal, shutil
import mss
import json, re, hashlib, uuid, getpass, platform, subprocess, requests, atexit, random
import time, win32con, win32api, pyperclip, threading, os, ctypes, pyautogui, sys, io
from pynput.keyboard import (
    Key, Controller as KeyboardController
)
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from datetime import datetime
from cryptography.fernet import Fernet

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QLabel, QPushButton, QGroupBox, QStatusBar, QMessageBox, QPlainTextEdit,
    QDialog, QProgressBar, QGraphicsDropShadowEffect, QDesktopWidget,QCheckBox,
    QComboBox, QShortcut, QRadioButton, QButtonGroup, QGridLayout, QGridLayout,
     QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import (
    QIcon, QFont, QFontDatabase, QRegion, QPixmap, QPainter, QDesktopServices, QIntValidator
    , QKeySequence
)
from PyQt5.QtCore import (
    QTimer, Qt, pyqtSignal, QObject, QRect, QSize, QUrl
)
from PyQt5.QtSvg import QSvgRenderer
from pyautogui import ImageNotFoundException
from collections import defaultdict
import cv2
import numpy as np

# ===================ALL OF FOLDER===================
FOLDER_IMG = "img"          #1
FOLDER_ROOM = "room"        #2
FOLDER_ICONS = "icons"      #3
FOLDER_MAIL = "mail"        #4
FOLDER_LOGIN = "login"      #5
FOLDER_CLAIM = "claim"      #6
FOLDER_GIFT = "gift"        #7  
FOLDER_AC = "AD"            #8

VERSION = "v1.0.0.4"  # Update this when building a new EXE

def check_for_updates():
    latest_release_api = "https://api.github.com/repos/Unknow-Hahaha/Pig/releases/latest"
    try:
        response = requests.get(latest_release_api, timeout=5)
        data = response.json()
        latest_version = data["tag_name"]

        if latest_version > VERSION:
            print(f"[UPDATE] New version {latest_version} available.")

            for asset in data["assets"]:
                if asset["name"].endswith(".exe"):
                    download_url = asset["browser_download_url"]
                    break
            else:
                print("[ERROR] No .exe found in release assets.")
                return

            # Handoff to Update.exe
            updater_path = resource_path("Update.exe")
            current_exe = sys.executable

            print(f"[OK] Launching updater: {updater_path}")
            subprocess.Popen([updater_path, download_url, current_exe])
            sys.exit()

        else:
            print("[✔] Already up to date.")

    except Exception as e:
        print(f"[ERROR] Update check failed: {e}")

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    # Relaunch the script with admin rights
    script = os.path.abspath(sys.argv[0])
    params = ' '.join([f'"{arg}"' for arg in sys.argv[1:]])
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
    sys.exit()

def delete_backup_if_exists():
    bak_path = sys.executable + ".bak"
    print(f"[DEBUG] Running from: {sys.executable}")
    print(f"[DEBUG] Looking for: {bak_path}")

    for attempt in range(5):
        try:
            if os.path.exists(bak_path):
                print(f"[DEBUG] Found .bak — deleting (attempt {attempt+1})")
                os.remove(bak_path)
                print(f"[CLEANUP] Deleted: {bak_path}")
                return
            else:
                print("[DEBUG] .bak file does not exist.")
                break
        except Exception as e:
            print(f"[WARN] Failed to delete .bak (attempt {attempt+1}): {e}")
            time.sleep(0.5)

    if os.path.exists(bak_path):
        print("[WARN] .bak file still exists after retries.")

delete_backup_if_exists()  # Call this right after your admin check

# Global Controllers
keyboard_controller = KeyboardController()
mouse_controller = MouseController()

# Flags
playback_flag = threading.Event()
stop_flag = threading.Event()
detect_death_error_logged = False
death_monitor_stopped_logged = False

# Flags
playback_flag = threading.Event()
stop_flag = threading.Event()
detect_death_error_logged = False
death_monitor_stopped_logged = False

try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass

# Optional: log screen info
screen_width, screen_height = pyautogui.size()
print(f"[INFO] Screen resolution: {screen_width}x{screen_height}")

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(__file__))

def resource_path(relative_path):
    """Get absolute path to resource, works for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(monitor))
    screen_color = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

def click_multiple_images(image_names, clicks=1, confidence=0.95):
    success = False
    for img_path in image_names:
        if os.path.exists(img_path):
            print(f"Checking image: {os.path.basename(img_path)}")
            clicked = click_image(img_path, clicks, confidence)
            if clicked:
                print(f"✅ Clicked on variation: {os.path.basename(img_path)}")
                success = True
        else:
            print(f"❌ Image path not found: {os.path.basename(img_path)}")
    
    if not success:
        print(f"❌ No image variation found for: {', '.join(image_names)}")
    return success

def click_image(img_path, clicks=1, confidence=0.97):
    """
    Detects and clicks up to `clicks` distinct matches of the image on screen.
    Avoids clicking the same location more than once.
    """
    if not os.path.exists(img_path):
        print(f"❌ Image path not found: {img_path}")
        return False

    template = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"❌ Failed to load image: {img_path}")
        return False

    screenshot = pyautogui.screenshot()
    screen_np = np.array(screenshot)
    screen_color = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

    template_h, template_w = template.shape[:2]
    matched_points = []
    clicked = 0

    res = cv2.matchTemplate(screen_color, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= confidence)

    seen = []

    for pt in zip(*loc[::-1]):
        center_x = pt[0] + template_w // 2
        center_y = pt[1] + template_h // 2

        # Avoid duplicate clicks near already clicked spots
        if any(abs(center_x - x) < template_w // 2 and abs(center_y - y) < template_h // 2 for x, y in seen):
            continue

        seen.append((center_x, center_y))

        pyautogui.moveTo(
            center_x + random.randint(-1, 1),
            center_y + random.randint(-1, 1),
            duration=random.uniform(0.01, 0.025)
        )
        time.sleep(random.uniform(0.005, 0.015))

        pyautogui.mouseDown()
        time.sleep(random.uniform(0.008, 0.02))
        pyautogui.mouseUp()
        time.sleep(random.uniform(0.02, 0.04))

        pyautogui.move(
            random.randint(-1, 1),
            random.randint(-1, 1),
            duration=random.uniform(0.01, 0.025)
        )

        clicked += 1
        print(f"✅ Clicked {os.path.basename(img_path)} at ({center_x},{center_y})")

        if clicked >= clicks:
            break

    if clicked == 0:
        print(f"❌ Could not find {os.path.basename(img_path)} with sufficient confidence")
        return False

    return True

clicked_once_images = set()

def click_once(img_path, **kwargs):
    if img_path in clicked_once_images:
        # print(f"⏭️ Skipped (already clicked once): {os.path.basename(img_path)}")
        return False
    clicked_once_images.add(img_path)
    return click_image(img_path, **kwargs)

def image_exists(img_path, confidence=0.9):
    screenshot = pyautogui.screenshot()
    screen_np = np.array(screenshot)
    screen_color = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

    template = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"❌ Failed to load image: {img_path}")
        return False

    res = cv2.matchTemplate(screen_color, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= confidence

def right_click_image(img_path, confidence=0.9):
    """
    Finds image on screen and right-clicks its center cleanly (no hold).
    """
    screenshot = pyautogui.screenshot()
    screen_np = np.array(screenshot)
    screen_color = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

    template = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"❌ Failed to load image: {img_path}")
        return False

    res = cv2.matchTemplate(screen_color, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= confidence:
        center_x = max_loc[0] + template.shape[1] // 2
        center_y = max_loc[1] + template.shape[0] // 2
        pyautogui.moveTo(center_x, center_y, duration=random.uniform(0.0008, 0.02))
        pyautogui.mouseDown(button='right')
        pyautogui.mouseUp(button='right')
        print(f"✅ Right-clicked {os.path.basename(img_path)}")
        return True
    else:
        print(f"❌ Could not find {os.path.basename(img_path)}")
        return False

def click_multiple_images(image_names, clicks=1, confidence=0.95):
    for img_path in image_names:
        if os.path.exists(img_path):
            print(f"Checking image: {os.path.basename(img_path)}")
            clicked = click_image(img_path, clicks, confidence)
            if clicked:
                print(f"✅ Clicked on variation: {os.path.basename(img_path)}")
                return True
        else:
            print(f"❌ Image path not found: {os.path.basename(img_path)}")
    
    print(f"❌ No image variation found for: {', '.join(image_names)}")
    return False

def human_like_click_image_auto(image_filename, confidence=0.85, double_click=False):
    image_path = resource_path(os.path.join(FOLDER_IMG, image_filename))
    for attempt in range(2):
        try:
            position = pyautogui.locateOnScreen(
                image_path,
                confidence=confidence,
                grayscale=True
            )
        except pyautogui.ImageNotFoundException:
            position = None

        if position:
            center = pyautogui.center(position)
            pyautogui.moveTo(
                center.x + random.randint(-1, 1),
                center.y + random.randint(-1, 1),
                duration=random.uniform(0.005, 0.015)
            )
            time.sleep(random.uniform(0.005, 0.015))
            for _ in range(2 if double_click else 1):
                pyautogui.mouseDown()
                time.sleep(random.uniform(0.008, 0.02))
                pyautogui.mouseUp()
                time.sleep(random.uniform(0.02, 0.04))
            pyautogui.move(random.randint(-1, 1), random.randint(-1, 1), duration=random.uniform(0.01, 0.025))
            return True
        time.sleep(random.uniform(0.05, 0.1))
    return False

def get_mouse_button_auto(button_str):
    return Button.left if "left" in button_str else Button.right

def get_pynput_key_auto(key_char):
    """Convert string key to pynput Key object when needed"""
    special_keys = {
        'space': Key.space,
        'esc': Key.esc,
        'enter': Key.enter,
        'tab': Key.tab,
        'backspace': Key.backspace,
        'shift': Key.shift,
        'ctrl': Key.ctrl,
        'alt': Key.alt,
        'caps_lock': Key.caps_lock,
        'cmd': Key.cmd,
    }
    return special_keys.get(key_char.lower(), key_char)

def hold_key_auto(key_char, duration):
    try:
        key = get_pynput_key_auto(key_char)
        keyboard_controller.press(key)
        time.sleep(duration)
        keyboard_controller.release(key)
    except Exception as e:
        print(f"[KeyError] {key_char}: {e}")

def hold_mouse(button_str, duration):
    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010

    button_str = str(button_str).lower()

    if "left" in button_str:
        down_flag = MOUSEEVENTF_LEFTDOWN
        up_flag = MOUSEEVENTF_LEFTUP
    elif "right" in button_str:
        down_flag = MOUSEEVENTF_RIGHTDOWN
        up_flag = MOUSEEVENTF_RIGHTUP
    else:
        print(f"[WARN] Unsupported mouse button: {button_str}")
        return

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        _anonymous_ = ("u",)
        _fields_ = [("type", ctypes.c_ulong), ("u", _INPUT)]

    def send_mouse_event(flag):
        mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=None)
        inp = INPUT(type=INPUT_MOUSE, mi=mi)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    send_mouse_event(down_flag)
    time.sleep(duration)
    send_mouse_event(up_flag)

def wait_and_click_auto(image_filename, label, confidence=0.85, interval=0.1):
    print(f"Waiting for {label}...")

    error_logged_precheck = False  # 👈 add this before the loop

    while not stop_flag.is_set():
        # ⬇️ PRE-CHECK: Skip clicking start.png if confirm.png is already visible
        if image_filename == "start.png":
            confirm_path = resource_path(os.path.join(FOLDER_IMG, "gun.png"))
            if not os.path.exists(confirm_path):
                if not error_logged_precheck:
                    print(f"[ERROR] confirm.png not found at {confirm_path}")
                    error_logged_precheck = True
            else:
                try:
                    confirm_visible = pyautogui.locateOnScreen(confirm_path, confidence=0.85)
                    if confirm_visible:
                        print("[SKIP] confirm.png already visible → skipping clicking start.png")
                        return True
                except Exception as e:
                    if not error_logged_precheck:
                        print(f"[ERROR] Pre-check for confirm.png: {e}")
                        error_logged_precheck = True

        # Attempt to click the target image
        clicked = human_like_click_image_auto(
            image_filename,
            confidence=confidence,
            double_click=(image_filename == "start.png")
        )
        if clicked:
            print(f"{label} clicked.")
            time.sleep(0.5)  # Let UI settle

            # ⬇️ POST-CHECK: After click, if confirm.png appears, skip re-click
            if image_filename == "start.png":
                try:
                    confirm_path = resource_path(os.path.join(FOLDER_IMG, "gun.png"))
                    confirm_visible = pyautogui.locateOnScreen(confirm_path, confidence=0.85)
                    if confirm_visible:
                        print("[SKIP] Detected confirm.png → skipping re-clicking start.png")
                        return True
                except Exception as e:
                    print(f"[ERROR] Checking for confirm.png: {e}")

            # ⬇️ Re-check if the target image is still visible
            try:
                position = pyautogui.locateOnScreen(
                    resource_path(os.path.join(FOLDER_IMG, image_filename)),
                    confidence=confidence,
                    grayscale=True
                )
                if position:
                    print(f"{label} still visible, clicking again...")
                    continue  # Retry
                else:
                    return
            except Exception as e:
                print(f"Error checking for {label}: {e}")
                return

        time.sleep(interval)

import cv2
import numpy as np
import itertools

def detect_spawn_auto(delay=0.05, confidence=0.9, scale_range=(0.98, 1.02), steps=2):
    print("🔍 Searching Map")

    # Optional: death check
    try:
        die_img = cv2.imread(resource_path(os.path.join(FOLDER_IMG, "die.png")), cv2.IMREAD_GRAYSCALE)
        screen = pyautogui.screenshot()
        screen_gray = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        if die_img is not None:
            res = cv2.matchTemplate(screen_gray, die_img, cv2.TM_CCOEFF_NORMED)
            if np.max(res) >= 0.9:
                print("💀 Death detected before spawn.")
                return None
    except Exception as e:
        print(f"[ERROR] Death check: {e}")

    # Load valid sn templates
    sn_templates = []
    for i in range(1, 10):
        path = resource_path(os.path.join(FOLDER_IMG, f"sn{i}.png"))
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sn_templates.append((i, img))

    if not sn_templates:
        print("[ERROR] No spawn images found.")
        return None

    cycle_sn = itertools.cycle(sn_templates)

    while not stop_flag.is_set():
        spawn_id, template = next(cycle_sn)

        try:
            screen = pyautogui.screenshot()
            screen_gray = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)

            for scale in np.linspace(scale_range[0], scale_range[1], steps):
                try:
                    resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)

                    if max_val >= confidence:
                        print(f"✅ Spawn detected: sn{spawn_id}.png (conf={max_val:.2f})")
                        return spawn_id
                except Exception:
                    continue
        except Exception as e:
            print(f"[ERROR] Screen check failed: {e}")

        time.sleep(delay)

    print("🛑 Stopped before spawn detection.")
    return None

def detect_death_auto(confidence=0.9, delay=0.5):
    global detect_death_error_logged, death_monitor_stopped_logged

    img_path = resource_path(os.path.join(FOLDER_IMG, "die.png"))
    confirmed = 0

    while not stop_flag.is_set():
        try:
            found = pyautogui.locateOnScreen(
                img_path, 
                confidence=confidence, 
                grayscale=True
            )
            if found:
                confirmed += 1
                if confirmed >= 2:
                    print("Death detected, Now Restart")
                    return True
            else:
                confirmed = 0

        except Exception as e:
            if not detect_death_error_logged:  # 👈 global check
                print(f"[ERROR] detect_death_auto: {e}")
                detect_death_error_logged = True

        time.sleep(delay)

    if not death_monitor_stopped_logged:
        print("[🛑] Death monitor stopped.")
        death_monitor_stopped_logged = True
    return False

def restart_game_auto():
    print("Restarting game...")

    keyboard_controller.press(Key.esc)
    time.sleep(0.2)

    for _ in range(2):
        success = human_like_click_image_auto("exit.png", confidence=0.85)
        if success:
            print(" Exit button clicked.")
            break
        print("Exit not found yet... retrying.")
        time.sleep(0.1)

    for _ in range(2):
        success = human_like_click_image_auto("confirm.png", confidence=0.85)
        if success:
            print("Confirm exit clicked.")
            break
        print("Confirm not found yet... retrying.")
        time.sleep(0.1)

    # Short buffer after confirm
    time.sleep(0.5)

# -----------------------------
# Main Action Functions
# -----------------------------
def group_by_time(actions, threshold=0.01):
    grouped = defaultdict(list)
    for action in sorted(actions, key=lambda a: a["start_time"]):
        bucket = round(action["start_time"] / threshold) * threshold
        grouped[bucket].append(action)
    return list(grouped.items())

death_thread = None
death_flag = threading.Event()

def replay_actions_for_spawn_auto(spawn_number):
    global death_thread
    died = [False]

    def monitor_death():
        print("[DEATH WATCH] Started")
        if detect_death_auto():
            died[0] = True
            death_flag.set()     # ✅ this now signals a death
            stop_flag.set()      # optional: still stop current replay
            print("[DEATH WATCH] Death detected — stopping playback")
            restart_game_auto()

    if not death_thread or not death_thread.is_alive():
        death_thread = threading.Thread(target=monitor_death, daemon=True)
        death_thread.start()

    if stop_flag.is_set():
        print("[REPLAY] Stop flag already set before starting replay. Exiting.")
        return

    filepath = resource_path(os.path.join(FOLDER_AC, f"sn{spawn_number}.json"))
    if not os.path.exists(filepath):
        print(f"[Missing] {filepath}")
        return

    with open(filepath, "r") as f:
        actions = json.load(f)

    # ✅ One-time death check before playback
    try:
        if pyautogui.locateOnScreen(resource_path(os.path.join(FOLDER_IMG, "die.png")), confidence=0.9, grayscale=True):
            print("💀 Death detected BEFORE playback. Attempting recovery...")
            start_wait = time.time()
            while not stop_flag.is_set() and (time.time() - start_wait) < 3:
                try:
                    if pyautogui.locateOnScreen(resource_path(os.path.join(FOLDER_IMG, "hp.png")), confidence=0.9, grayscale=False):
                        print("[OK] HP restored. Restarting game.")
                        stop_flag.clear()
                        restart_game_auto()
                        return
                except Exception as e:
                    print(f"[HP Check Error] {e}")
                time.sleep(0.2)

            print("[FAIL] HP not found in 3s — forcing restart.")
            restart_game_auto()
            return
    except Exception as e:
        print(f"[Death Check Error] {e}")

    start_time = time.time()
    grouped_actions = group_by_time(actions)

    for group_time, group in grouped_actions:
        if stop_flag.is_set():
            print("[STOP] Playback aborted.")
            return

        wait = group_time - (time.time() - start_time)
        if wait > 0:
            time.sleep(wait)

        for action in group:
            if stop_flag.is_set():
                print("[REPLAY] Stop flag mid-group — exiting.")
                return

            typ = action["type"]

            if typ == "key_hold":
                k = action["key"]
                d = action["duration"]
                threading.Thread(target=hold_key_auto, args=(k, d), daemon=True).start()

            elif typ == "mouse_hold":
                b = action["button"]
                d = action["duration"]
                threading.Thread(target=hold_mouse, args=(b, d), daemon=True).start()


    print(f"[✅] Finished running: {filepath}")

    # # 🧠 Final death check: start.png must appear
    # print("[INFO] Waiting for start.png to confirm survival...")

    # wait_start_timeout = 20  # seconds
    # found_start = False
    # error_logged = False  # 👈 prevent spam

    # for _ in range(int(wait_start_timeout / 0.25)):
    #     try:
    #         if pyautogui.locateOnScreen(
    #             resource_path(os.path.join(FOLDER_IMG, "start.png")), 
    #             confidence=0.85, 
    #             grayscale=False
    #             ):
    #             found_start = True
    #             print("[OK] start.png appeared — survived.")
    #             break
    
    #     except Exception as e:
    #         if not error_logged:
    #             print(f"[ERROR] checking for start.png: {e}")
    #             error_logged = True  # 👈 log once only
    #     time.sleep(0.5)

    # if not found_start:
    #     print("[☠] start.png not found after replay — assumed death. Restarting...")
    #     restart_game_auto()

def auto_sequence_loop_auto():
    while not stop_flag.is_set():
        
        wait_and_click_auto("start.png", "Start Button")
        if stop_flag.is_set():
            break

        wait_and_click_auto("confirm.png", "Confirm Button")
        if stop_flag.is_set():
            break

        spawn = detect_spawn_auto()
        if stop_flag.is_set():
            break

        if spawn:
            replay_actions_for_spawn_auto(spawn)
            print("[*] Waiting to restart after replay...\n")
            if stop_flag.is_set():
                restart_game_auto()
                stop_flag.clear()
                continue

# -----------------------------
# Control Functions
# -----------------------------

def on_press_auto(key):
    if key == keyboard.Key.f3:
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for widget in app.topLevelWidgets():
                    if isinstance(widget, ReplayBotGUI):  # <- ensure it's your main window
                        print("[F3] Calling toggle_bot() on ReplayBotGUI")
                        widget.toggle_bot()
                        return
        except Exception as e:
            print(f"[F3 ERROR] Failed to trigger GUI toggle_bot: {e}")

def start_replay_bot_auto():
    if not playback_flag.is_set():
        playback_flag.set()
        stop_flag.clear()  # ensure previous stop doesn't block new loop
        threading.Thread(target=auto_sequence_loop_auto, daemon=True).start()

def run_single_match_auto():
    playback_flag.clear()  # prevent any auto-loop leftovers
    # stop_flag.clear()      # ensure nothing is stuck from previous death

    wait_and_click_auto("start.png", "Start Button")
    wait_and_click_auto("confirm.png", "Confirm Button")

    spawn = detect_spawn_auto()
    if spawn:
        replay_actions_for_spawn_auto(spawn)
        if stop_flag.is_set():
            print("[REPLAY] Match interrupted by death — stopped and exited.")
            # stop_flag.clear()
            return
        print("[OK] Finished 1 match replay.")

def stop_replay_bot_auto():
    stop_flag.set()

# =============================
# ======== Mail Logicc ========
# =============================

running = False
already_clicked = set()
def process_mail_sequence(skip_start_check=False):
    clicked_once_images.clear()
    start_image = resource_path(os.path.join(FOLDER_IMG, "start.png"))
    mail_clicked = False

    if not skip_start_check:
        print("[INFO] Waiting for start.png to appear before starting mail process...")
        while True:
            try:
                if pyautogui.locateOnScreen(start_image, confidence=0.85, grayscale=False):
                    break
            except Exception:
                pass
        print("[OK] start.png detected — clicking mail1 until mail2 or mail4 appears...")
    else:
        print("[SKIP] Skipping start.png check as instructed (Clear Mail mode).")

    while not image_exists(resource_path(os.path.join(FOLDER_GIFT, "mail.png"))):
        click_image(resource_path(os.path.join(FOLDER_MAIL, "mail1.png")), clicks=10)

    for i in range(75):
        if image_exists(resource_path(os.path.join(FOLDER_GIFT, "end2.png"))):
            print("🛑 Detected end.png — stopping loop.")
            click_image(resource_path(os.path.join(FOLDER_GIFT, "mail3.png")))
            click_image(resource_path(os.path.join(FOLDER_GIFT, "delete.png")), confidence=0.9)
            click_image(resource_path(os.path.join(FOLDER_GIFT, "mail_confirm.png")), confidence=0.9)
            print("⏳ Waiting for wait.png to disappear...")
            while image_exists(resource_path(os.path.join(FOLDER_GIFT, "wait.png"))):
                time.sleep(1)
            pyautogui.press('esc')
            break

        if not mail_clicked: 
            mail_clicked = click_multiple_images([
                resource_path(os.path.join(FOLDER_GIFT, "mail2.png")),
                resource_path(os.path.join(FOLDER_GIFT, "mail4.png"))
            ], clicks=1, confidence=0.85)

        click_image(resource_path(os.path.join(FOLDER_GIFT, "mail3.png")))

        for _ in range(7):
            win32api.keybd_event(win32con.VK_DOWN, 0, 0, 0)
            win32api.keybd_event(win32con.VK_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)

        print(f"[{i+1}/75] Round completed")
 
    # print("[INFO] Clicking mail3.png one last time before delete...")
    # try:
    #     final_mail3 = pyautogui.locateOnScreen(mail3_path, confidence=0.85, grayscale=False)
    #     if final_mail3:
    #         center = pyautogui.center(final_mail3)
    #         click_image(center.x, center.y)
    #         print("[OK] Final mail3.png clicked.")
    #     else:
    #         print("[WARN] mail3.png not found for final click.")
    # except Exception as e:
    #     print(f"[ERROR] During final mail3.png click: {e}")

    # print("[INFO] Attempting delete...")
    # found = pyautogui.locateOnScreen(delete_path, confidence=0.85, grayscale=False)
    # if found:
    #     center = pyautogui.center(found)
    #     click_image(center.x, center.y)
    #     pyautogui.press("enter")
    #     time.sleep(2)
    #     pyautogui.press("esc")

    #     print("[OK] Deletion confirmed.")
    # else:
    #     print("[WARN] delete.png not found. Skipping delete.")

# Login Logic
def fast_click_login(image_path, clicks=1):
    try:
        position = pyautogui.locateOnScreen(image_path, confidence=0.85, grayscale=True)
        if position:
            center = pyautogui.center(position)
            pyautogui.moveTo(
                center.x + random.randint(-1, 1),
                center.y + random.randint(-1, 1),
                duration=0.01
            )
            time.sleep(0.01)
            for _ in range(clicks):
                pyautogui.mouseDown()
                time.sleep(0.008)
                pyautogui.mouseUp()
                time.sleep(0.01)
            pyautogui.move(random.randint(-1, 1), random.randint(-1, 1), duration=0.005)
            print(f"Clicked {os.path.basename(image_path)} ({clicks}x)")
            return True
        else:
            print(f"{os.path.basename(image_path)} not found")
            return False
    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return False

def fast_double_click_and_paste_login(image_path, text):
    try:
        position = pyautogui.locateOnScreen(image_path, confidence=0.85, grayscale=True)
        if position:
            center = pyautogui.center(position)
            pyautogui.moveTo(center.x, center.y, duration=0.01)
            for _ in range(2):
                pyautogui.mouseDown()
                time.sleep(0.01)
                pyautogui.mouseUp()
                time.sleep(0.01)
            pyperclip.copy(text)
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'v')
            print(f"Double-clicked {os.path.basename(image_path)} and pasted text")
            return True
        else:
            print(f"{os.path.basename(image_path)} not found")
            return False
    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return False

def click_images_login(mode):
    global running, already_clicked
    running = True
    already_clicked.clear()

    mode_image = "5.png" if mode == "Crazy" else "9.png"

    actions = [
        ("1.png", 10),
        ("2.png", 1),
        ("3.png", 1),
        ("4.png", 3),
        (mode_image, 1),
        ("6.png", 1),  # ← Use image based on ComboBox
        ("7.png", "paste"),
        ("8.png", 1)
    ]

    for index, (image_name, action) in enumerate(actions):
        if not running:
            break
        if image_name in already_clicked:
            continue

        path = resource_path(os.path.join(FOLDER_ROOM, image_name))

        if action == "paste":
            if fast_double_click_and_paste_login(path, "Naalonh_0886016614"):
                already_clicked.add(image_name)
                time.sleep(0.5)
        else:
            if fast_click_login(path, action):
                already_clicked.add(image_name)

        time.sleep(0.1)

def start_room_bot_login(mode):
    global running, already_clicked
    running = True
    already_clicked.clear()
    print("[INFO] Starting Room Bot clicks...")
    threading.Thread(target=click_images_login, args=(mode,), daemon=True).start()

# ------------------- LOGIN BOT GUI -------------------
def strong_human_click_login(image_file):
    path = resource_path(os.path.join(FOLDER_LOGIN, image_file))
    try:
        position = pyautogui.locateCenterOnScreen(
            path, 
            confidence=0.80, 
            grayscale=True
        )
        if position:
            center = position
            pyautogui.moveTo(
                center.x + random.randint(-1, 1), 
                center.y + random.randint(-1, 1), 
                duration=random.uniform(0.02, 0.05), 
                tween=pyautogui.easeOutQuad
            )
            for _ in range(2):
                pyautogui.mouseDown()
                time.sleep(random.uniform(0.01, 0.025))
                pyautogui.mouseUp()
                time.sleep(random.uniform(0.03, 0.05))
            print(f"[OK] Double-clicked: {image_file}")
            return True
        else:
            print(f"[ ] Not found: {image_file}")
            return False
    except Exception as e:
        print(f"[ERROR] Error on {image_file}: {e}")
        return False

def open_saved_app_login():
    if os.path.exists("app_path.txt"):
        try:
            with open("app_path.txt", "r") as f:
                path = f.read().strip()
                if os.path.exists(path):
                    subprocess.Popen(path)
                    print(f"[OK] Opened saved app: {path}")
                else:
                    print(f"[ERROR] App not found at saved path: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to open app: {e}")
            
def click_start_and_wait_for_1():
    print("[INFO] Checking if 1.png is already visible...")

    image_path_1 = resource_path(os.path.join(FOLDER_LOGIN, "1.png"))
    try:
        position = pyautogui.locateCenterOnScreen(image_path_1, confidence=0.85, grayscale=False)
        if position:
            print("[OK] 1.png already visible on screen — skipping start.png.")
            pyautogui.moveTo(position.x + random.randint(-1, 1), position.y + random.randint(-1, 1),
                             duration=random.uniform(0.02, 0.05), tween=pyautogui.easeOutQuad)
            for _ in range(2):
                pyautogui.mouseDown()
                time.sleep(random.uniform(0.01, 0.025))
                pyautogui.mouseUp()
                time.sleep(random.uniform(0.03, 0.05))
            return True  # ✅ ADD THIS to exit earlyERROR
    except Exception as e:
        print(f"[ERROR] Error while checking for 1.png: {e}")

def wait_and_double_click_login_start(delay=0.00001):
    print("[WAIT] Waiting for login/start.png...")

    path = resource_path(os.path.join(FOLDER_LOGIN, "start.png"))

    while True:
        try:
            position = pyautogui.locateCenterOnScreen(path, confidence=0.85, grayscale=False)
            if position:
                pyautogui.moveTo(
                    position.x + random.randint(-1, 1),
                    position.y + random.randint(-1, 1),
                    duration=random.uniform(0.02, 0.04),
                    tween=pyautogui.easeOutQuad
                )
                for _ in range(2):
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()
                    time.sleep(0.01)

                print("[OK] Double-clicked login/start.png")
                return True

        except ImageNotFoundException:
            pass  # Just means image not found — not an error
        except Exception as e:
            print(f"[ERROR] wait_and_double_click_login_start: {repr(e)}")

        time.sleep(delay)

def wait_and_double_click_login(image_file, delay=0.00001):
    print(f"[WAIT] Waiting for {image_file} (no confirm_loops)...")

    path = resource_path(os.path.join(FOLDER_LOGIN, image_file))

    while True:
        try:
            position = pyautogui.locateCenterOnScreen(path, confidence=0.85, grayscale=False)
            if position:
                center = position
                pyautogui.moveTo(
                    center.x + random.randint(-1, 1),
                    center.y + random.randint(-1, 1),
                    duration=random.uniform(0.02, 0.04),
                    tween=pyautogui.easeOutQuad
                )
                for _ in range(2):
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()
                    time.sleep(0.01)

                print(f"[OK] Double-clicked: {image_file}")
                return True

        except ImageNotFoundException:
            pass  # just ignore if not found
        except Exception as e:
            print(f"[ERROR] wait_and_double_click_login: {repr(e)}")

        time.sleep(delay)

def click_login_start_once():
    image_file = "start.png"
    path = resource_path(os.path.join(FOLDER_LOGIN, image_file))

    try:
        position = pyautogui.locateCenterOnScreen(path, confidence=0.85, grayscale=False)
        if position:
            pyautogui.moveTo(
                position.x + random.randint(-1, 1),
                position.y + random.randint(-1, 1),
                duration=random.uniform(0.02, 0.05),
                tween=pyautogui.easeOutQuad
            )
            for _ in range(2):  # double-click
                pyautogui.mouseDown()
                time.sleep(0.01)
                pyautogui.mouseUp()
                time.sleep(0.03)
            print("[OK] Clicked login/start.png")
            return True
        else:
            print("[MISS] login/start.png not found")
            return False

    except Exception as e:
        print(f"[ERROR] click_login_start_once: {repr(e)}")
        return False

def quick_click_group(image_file, timeout=3):
    if not image_file:
        print(f"[WARN] No image file specified for group click.")
        return False

    path = resource_path(os.path.join(FOLDER_LOGIN, image_file))
    start_time = time.time()

    while True:
        try:
            position = pyautogui.locateCenterOnScreen(path, confidence=0.85, grayscale=True)
        except pyautogui.ImageNotFoundException:
            position = None
        except Exception as e:
            print(f"[ERROR] quick_click_group: {e}")
            position = None

        if position:
            center = position
            pyautogui.moveTo(center.x, center.y, duration=0.03)
            for _ in range(2):
                pyautogui.mouseDown()
                time.sleep(0.01)
                pyautogui.mouseUp()
                time.sleep(0.01)
            print(f"[OK] Clicked group image: {image_file}")
            return True

        if time.time() - start_time > timeout:
            print(f"[TIMEOUT] {image_file} not found in {timeout}s")
            return False

        time.sleep(0.2)

def human_like_paste_login(text):
    pyperclip.copy(text)
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.2)
    
class LoginBotGUI(QMainWindow):
    def __init__(self, mode="Crazy"):
        super().__init__()
        self.mode = mode
        self.setWindowTitle("Login Bot")
        self.setWindowIcon(QIcon(resource_path("logo.png")))

        self.setGeometry(200, 200, 400, 300)

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.remember_checkbox = QCheckBox("Remember Me")
        
        self.group1_buttons = QButtonGroup()
        self.group2_buttons = QButtonGroup()
        self.group1_key_map = {}
        self.group2_key_map = {}

        self.init_ui_login()
        self.load_credentials_login()

    CONFIG_FILE = "config.json"

    def auto_save_all(self):
        self.save_credentials(
            self.username_input.text(),
            self.password_input.text()
        )
        print("[OK] Auto-saved radio button selection")

    def save_credentials(self, username, password):
        data = {
            "remember": self.remember_checkbox.isChecked(),
            "username": username,
            "password": password,
            "group1_id": self.group1_buttons.checkedId(),
            "group2_id": self.group2_buttons.checkedId()
        }
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(data, f)

    def load_credentials_login(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as f:
                data = json.load(f)
                if data.get("remember"):
                    self.username_input.setText(data.get("username", ""))
                    self.password_input.setText(data.get("password", ""))
                    self.remember_checkbox.setChecked(True)

                # Restore radio button states
                group1_id = data.get("group1_id")
                group2_id = data.get("group2_id")
                if group1_id is not None:
                    button1 = self.group1_buttons.button(group1_id)
                    if button1:
                        button1.setChecked(True)
                if group2_id is not None:
                    button2 = self.group2_buttons.button(group2_id)
                    if button2:
                        button2.setChecked(True)

    def manual_save_credentials_login(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select App File to Auto-Open", "", "Executable Files (*.exe);;All Files (*.*)")
        if file_path:
            with open("app_path.txt", "w") as f:
                f.write(file_path)
            print(f"[OK] App path saved: {file_path}")

    def init_ui_login(self):
        layout = QVBoxLayout()

        save_button = QPushButton("Save File")
        save_button.clicked.connect(self.manual_save_credentials_login)
        layout.addWidget(save_button)

        form = QGridLayout()
        form.addWidget(QLabel("Username:"), 0, 0)
        form.addWidget(self.username_input, 0, 1)
        form.addWidget(QLabel("Password:"), 1, 0)
        form.addWidget(self.password_input, 1, 1)
        form.addWidget(self.remember_checkbox, 2, 1)
        layout.addLayout(form)

        # Server Block 1
        block1_group = QGroupBox("Server ទី​ 1:")
        block1_layout = QHBoxLayout()
        block1_options = {
            "ti1": "ទី 1",
            "ti3": "ទី 3"
        }
        for i, (key, label) in enumerate(block1_options.items()):
            btn = QRadioButton(label)
            self.group1_buttons.addButton(btn)
            self.group1_buttons.setId(btn, i)
            self.group1_key_map[i] = key
            block1_layout.addWidget(btn)
        block1_group.setLayout(block1_layout)
        layout.addWidget(block1_group)

        # Server Block 2
        block2_group = QGroupBox("Server ទី​ 2:")
        block2_layout = QGridLayout()
        block2_options = {
            "tii1": "ទី 1",
            "tii2": "ទី 2",
            "tii3": "ទី 3",
            "tii4": "ទី 4",
            "tii5": "ទី 5",
            "tii6": "ទី 6"
        }
        for i, (key, label) in enumerate(block2_options.items()):
            btn = QRadioButton(label)
            self.group2_buttons.addButton(btn)
            self.group2_buttons.setId(btn, i)
            self.group2_key_map[i] = key
            block2_layout.addWidget(btn, i // 2, i % 2)
        block2_group.setLayout(block2_layout)
        layout.addWidget(block2_group)
        
        # Connect radio selections to auto-save
        self.group1_buttons.buttonClicked.connect(self.auto_save_all)
        self.group2_buttons.buttonClicked.connect(self.auto_save_all)

        save_account_button = QPushButton("Save Account")
        save_account_button.clicked.connect(self.save_account_only_login)
        layout.addWidget(save_account_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def save_account_only_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        self.save_credentials(username, password)
        print("[OK] Account credentials saved.")
        self.close()

    def run_sequence_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        self.save_credentials(username, password)

        group1_id = self.group1_buttons.checkedId()
        group2_id = self.group2_buttons.checkedId()

        group1 = self.group1_key_map.get(group1_id, "")
        group2 = self.group2_key_map.get(group2_id, "")

        threading.Thread(target=self.run_login_sequence, args=(username, password, group1, group2), daemon=True).start()
    
    def room_image_visible_login(self):
        try:
            image_path = resource_path(os.path.join("room", "1.png"))
            position = pyautogui.locateOnScreen(image_path, confidence=0.85, grayscale=True)
            return position is not None
        except Exception as e:
            print(f"[ERROR] Error while checking for room/1.png: {e}")
            return False
        
    def run_login_sequence(self, username, password, group1, group2):
        try:
            # ✅ Skip full login if already on start screen
            try:
                if pyautogui.locateOnScreen(resource_path(os.path.join("img", "start.png")), confidence=0.85, grayscale=True):
                    print("[SKIP] Detected start.png immediately — skipping login sequence.")
                    return
            except Exception as e:
                print(f"[ERROR] checking for start.png: {e}")

            open_saved_app_login()
            time.sleep(1)

            if self.room_image_visible_login():
                print("[OK] room/1.png detected. Skipping login images.")
                open_saved_app_login()
                time.sleep(1)
            else:
                clicked_early = click_start_and_wait_for_1()
                if not clicked_early:
                    wait_and_double_click_login_start()
                    wait_and_double_click_login("1.png", delay=0.00005)

                human_like_paste_login(username)
                pyautogui.press("tab")
                time.sleep(0.2)

                human_like_paste_login(password)
                pyautogui.press("enter")
                time.sleep(0.2)

                if group1:
                    quick_click_group(f"{group1}.png")
                if group2:
                    quick_click_group(f"{group2}.png")

            print("[INFO] Login completed. Waiting 2s before Room Bot starts...")
            time.sleep(2)
            click_images_login(self.mode)

        except Exception as e:
            print(f"[EXCEPTION] Login sequence crashed: {e}")

#   Naalonh

# Set the icon using internal path

def svg_to_icon(path, size=20):
    if not os.path.exists(path):
        print(f"[SVG ERROR] File not found: {path}")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        print(f"[SVG ERROR] Renderer failed to load: {path}")
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)

# --- LICENSE SETUP ---
FERNET_KEY = b'Nl6uzEuVO9bZ1TqMskpxZids9qaO3WypjvLmL_frGgo='
cipher = Fernet(FERNET_KEY)
LOCAL_LICENSE_CACHE = ".env"
LICENSE_URL = "https://gist.githubusercontent.com/Unknow-Hahaha/68db07fa5c89c058db0565ae9c72c134/raw/license_pig.txt"

def save_license_to_cache(valid_block):
    encrypted = cipher.encrypt(valid_block.encode())
    with open(".env", "wb") as f:
        f.write(encrypted)

def load_license_from_cache():
    if os.path.exists(".env"):
        try:
            with open(".env", "rb") as f:  # ✅ Correct filename
                encrypted = f.read()
            decrypted = cipher.decrypt(encrypted).decode()
            return decrypted
        except Exception as e:
            print(f"[Cache Read Error] {e}")
            return None
    return None

def check_license():
    """Check license validity and cache only the relevant block to .dat file."""
    try:
        print("Attempting to fetch fresh license...")
        fresh_license = fetch_license_from_gist()

        if fresh_license:
            print("Fetched license content:", fresh_license)
            valid_block = is_license_valid(fresh_license)

            if valid_block:
                print("License is valid. Writing to cache...")
                try:
                    with open(LOCAL_LICENSE_CACHE, 'w') as f:
                        f.write(valid_block)
                    print(f"Successfully wrote to {LOCAL_LICENSE_CACHE}")
                    return True
                except Exception as e:
                    print(f"Error writing cache file: {e}")
            else:
                print("License validation failed from Gist.")
                # ❗ DELETE OLD CACHE IF NEW LICENSE SAYS YOU'RE INVALID
                if os.path.exists(LOCAL_LICENSE_CACHE):
                    print("Removing invalid cache file...")
                    os.remove(LOCAL_LICENSE_CACHE)

        else:
            print("Failed to fetch fresh license.")

        # Fallback to cached file
        if os.path.exists(LOCAL_LICENSE_CACHE):
            os.remove(LOCAL_LICENSE_CACHE)
            print("Checking cached license...")
            try:
                with open(LOCAL_LICENSE_CACHE, 'r') as f:
                    cached_license = f.read()
                    if is_license_valid(cached_license):
                        print("Using valid cached license")
                        return True
                    else:
                        print("Cached license invalid. Deleting cache.")
                        os.remove(LOCAL_LICENSE_CACHE)
            except Exception as e:
                print(f"Error reading cache: {e}")
                os.remove(LOCAL_LICENSE_CACHE)

        print("No valid license available.")
        return False

    except Exception as e:
        print(f"License check error: {e}")
        return False

def get_device_fingerprint():
    try:
        components = []

        # Get Machine GUID (Windows only)
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
                machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                components.append(machine_guid)
            except Exception as e:
                print(f"[HWID] MachineGuid error: {e}")

        # Volume Serial Number (C drive)
        try:
            vol = subprocess.check_output("vol C:", shell=True).decode(errors="ignore")
            serial = re.search(r"([A-F0-9\-]{4,})", vol)
            if serial:
                components.append(serial.group(1))
        except Exception as e:
            print(f"[HWID] Volume error: {e}")

        # Username
        try:
            components.append(getpass.getuser())
        except:
            pass

        # Fallback to MAC
        if not components:
            components.append(str(uuid.getnode()))

        # Join and hash with MD5 (32 characters)
        raw = "|".join(components).encode()
        return hashlib.md5(raw).hexdigest().upper()

    except Exception as e:
        print(f"[HWID Error] {e}")
        return hashlib.md5(b"default").hexdigest().upper()

def get_latest_license_url():
    """Safely get the latest license URL with multiple fallback options"""
    try:
        # First try to read from local license_pig.txt if exists
        if os.path.exists("license_pig.txt"):
            print("Removing local license_pig.txt for clean fetch next time...")
            os.remove("license_pig.txt")
            with open("license_pig.txt", "r") as f:
                content = f.read()
                if content.strip():  # Only return if file has content
                    print("Using local license_pig.txt")
                    return None  # Return None to indicate using local file

        # If no local file, try GitHub API with authentication
        gist_api_url = "https://api.github.com/gists/68db07fa5c89c058db0565ae9c72c134"
        headers = {
            "User-Agent": "NaaAuto-License-Checker/1.0",
            "Accept": "application/vnd.github.v3+json",
            # "Authorization": "token ghp_your_token_here"  # Replace with your GitHub token
        }
        
        response = requests.get(gist_api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "history" in data and len(data["history"]) > 0:
            url = f"https://gist.githubusercontent.com/Unknow-Hahaha/68db07fa5c89c058db0565ae9c72c134/raw/{data['history'][0]['version']}/license_pig.txt"
            print(f"Using GitHub versioned URL: {url}")
            return url
        
        if "updated_at" in data:
            url = f"https://gist.githubusercontent.com/Unknow-Hahaha/68db07fa5c89c058db0565ae9c72c134/raw/?ts={int(time.time())}"
            print(f"Using GitHub timestamp URL: {url}")
            return url
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("GitHub rate limit exceeded, falling back to cached URL")
        else:
            print(f"HTTP error fetching license URL: {e}")
    except Exception as e:
        print(f"Error fetching license URL: {e}")

    # Final fallback
    url = f"https://gist.githubusercontent.com/Unknow-Hahaha/68db07fa5c89c058db0565ae9c72c134/raw/license_pig.txt?ts={int(time.time())}"
    print(f"Using fallback URL: {url}")
    return url

def fetch_license_from_gist():
    """Fetch the license from GitHub Gist or local file"""
    url = get_latest_license_url()
    if url is None:  # Using local file
        try:
            with open("license_pig.txt", "r") as f:
                return f.read().strip()
        except Exception:
            return None
            
    try:
        headers = {"User-Agent": "NaaAuto-License-Checker/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except Exception:
        return None

def is_license_valid(license_content):
    if not license_content:
        return None  # Changed to return None instead of False

    try:
        device_hash = get_device_fingerprint()
        revoked_hashes = []
        
        # Split license content into individual device blocks
        device_blocks = license_content.split('\n\n')

        for block in device_blocks:
            license_data = {}
            block_lines = []
            
            for line in block.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                block_lines.append(line)  # Keep original lines
                if line.startswith('REVOKED:'):
                    revoked_hashes.append(line[8:].strip())
                elif '=' in line:
                    key, value = line.split('=', 1)
                    license_data[key.strip()] = value.strip()

            # Check if this block matches our device
            if license_data.get('DEVICE_HASH') == device_hash:
                # Check expiry if present
                if 'EXPIRY' in license_data:
                    try:
                        expiry = datetime.strptime(license_data['EXPIRY'], "%Y-%m-%d %H:%M")
                    except ValueError:
                        try:
                            expiry = datetime.strptime(license_data['EXPIRY'], "%Y-%m-%d").replace(hour=23, minute=59, second=59)
                        except ValueError:
                            continue  # Skip invalid expiry formats
                    
                    if datetime.now() <= expiry:
                        return '\n'.join(block_lines)  # Return the valid block
                else:
                    return '\n'.join(block_lines)  # No expiry = valid

        return None  # No matching device found

    except Exception as e:
        print(f"Validation error: {e}")
        return None

def get_current_user_info():
    device_hash = get_device_fingerprint()
    license_text = load_license_from_cache() or fetch_license_from_gist()

    if not license_text:
        return "Unknown", "Unknown"

    blocks = license_text.split("\n\n")
    for block in blocks:
        user = "Unknown"
        expiry_str = ""
        matched = False

        for line in block.strip().splitlines():
            if line.startswith("DEVICE_HASH=") and device_hash in line:
                matched = True
            elif line.startswith("USER="):
                user = line.split("=", 1)[1].strip()
            elif line.startswith("EXPIRY="):
                expiry_str = line.split("=", 1)[1].strip()

        if matched:
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M")
            except ValueError:
                try:
                    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
                except ValueError:
                    expiry_dt = None

            if expiry_dt:
                now = datetime.now()
                days_left = (expiry_dt - now).days
                if days_left >= 0:
                    return user, f"{expiry_dt.strftime('%Y-%m-%d')} | {days_left} days left"
                else:
                    return user, f"{expiry_dt.strftime('%Y-%m-%d')} | ❌ Expired"
            else:
                return user, f"{expiry_str} | Invalid Date"

    return "Unknown", "Unknown"

def delete_license_cache():
    try:
        if os.path.exists(LOCAL_LICENSE_CACHE):
            os.remove(LOCAL_LICENSE_CACHE)
            print("🧹 Deleted .env on exit.")
    except Exception as e:
        print(f"[Exit Cleanup Error] {e}")

atexit.register(delete_license_cache)

from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QRectF

class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setChecked(False)
        self.stateChanged.connect(self.update)

    def paintEvent(self, event):
        radius = 14
        circle_diameter = 24
        margin = (self.height() - circle_diameter) // 2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        if self.isChecked():
            bg_color = QColor("#3ba9fc")
        else:
            bg_color = QColor("#404040")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), radius, radius)

        # Circle (thumb)
        if self.isChecked():
            circle_x = self.width() - circle_diameter - margin
            circle_color = QColor("black")
        else:
            circle_x = margin
            circle_color = QColor("#d3d3d3")

        circle_rect = QRectF(circle_x, margin, circle_diameter, circle_diameter)
        painter.setBrush(QBrush(circle_color))
        painter.drawEllipse(circle_rect)
        painter.end()

    def style(self, checked):
        if checked:
            return """
            QCheckBox::indicator { width: 0; height: 0; }
            QCheckBox {
                background-color: #3ba9fc;
                border-radius: 14px;
                padding: 0px;
            }
            QCheckBox::before {
                content: '';
            }
            QCheckBox::after {
                content: '';
            }
            QCheckBox {
                qproperty-text: '';
            }
            QCheckBox::indicator {
                width: 0;
                height: 0;
            }
            QCheckBox::indicator:checked {
                background-color: transparent;
            }
            QCheckBox::indicator {
                background-color: transparent;
            }
            QCheckBox::indicator:checked::after {
                background-color: black;
            }
            QCheckBox::indicator::after {
                content: '';
                position: absolute;
                left: 28px;
                top: 2px;
                width: 24px;
                height: 24px;
                border-radius: 12px;
                background-color: black;
            }
            """
        else:
            return """
            QCheckBox::indicator { width: 0; height: 0; }
            QCheckBox {
                background-color: #404040;
                border-radius: 14px;
                padding: 0px;
            }
            QCheckBox::before {
                content: '';
            }
            QCheckBox::after {
                content: '';
            }
            QCheckBox {
                qproperty-text: '';
            }
            QCheckBox::indicator {
                width: 0;
                height: 0;
            }
            QCheckBox::indicator:checked {
                background-color: transparent;
            }
            QCheckBox::indicator {
                background-color: transparent;
            }
            QCheckBox::indicator::after {
                content: '';
                position: absolute;
                left: 2px;
                top: 2px;
                width: 24px;
                height: 24px;
                border-radius: 12px;
                background-color: #d3d3d3;
            }
            """

class LicenseCheckingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Naalonh | Checking License")
        self.setFixedSize(400, 170)
        self.setWindowIcon(QIcon(resource_path("logo.png")))

        self.setModal(True)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        self.layout = QVBoxLayout(self)

        self.label = QLabel("🔐 Checking License...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(18)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #0d47a1;
                border-radius: 4px;
                background-color: #e3f2fd;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        self.layout.addWidget(self.progress)

        self.progress_value = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_animation)

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.animate_progress_bar)


        self.status = QLabel("Please wait while we validate your license.")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFont(QFont("Segoe UI", 9))
        self.layout.addWidget(self.status)

        self.button_row = QHBoxLayout()

        self.send_channel_button = QPushButton("Group Tool")
        self.send_channel_button.setVisible(False)
        self.send_channel_button.setStyleSheet("""
        QPushButton
            { 
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
            font-size: 9pt;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.send_channel_button.clicked.connect(self.send_to_channel)
        self.button_row.addWidget(self.send_channel_button)

        self.send_me_button = QPushButton("Send ID to Naalonh")
        
        self.send_me_button.setVisible(False)
        self.send_me_button.setStyleSheet("""
        QPushButton
            { 
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
            font-size: 9pt;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.send_me_button.clicked.connect(self.send_to_me)

        self.copy_id_button = QPushButton("Copy ID")
        self.copy_id_button.setVisible(False)
        self.copy_id_button.setStyleSheet("""
        QPushButton
            { 
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
            font-size: 9pt;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.copy_id_button.clicked.connect(self.copy_device_id)
        self.button_row.addWidget(self.copy_id_button)

        self.button_row.addWidget(self.send_me_button)

        self.layout.addLayout(self.button_row)

        self.result = None
        QTimer.singleShot(100, self.run_check)

        self.spinner_frames = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
        self.spinner_index = 0
        self.dot_cycle = ["", ".", "..", "..."]
        self.dot_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.animate_spinner)

    def animate_progress_bar(self):
        current = self.progress.value()
        if current < 99:
            self.progress.setValue(current + 1)
        else:
            self.progress_timer.stop()

    def copy_device_id(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.device_id)
        QMessageBox.information(self, "COPY ID | Naalonh", "Copy hx sent jol gp tv bro")

    def send_to_channel(self):
        msg = f"License Request\nDevice ID: {self.device_id}"
        link = f"https://t.me/NaalonhTools?url=&text={requests.utils.quote(msg)}"
        QDesktopServices.openUrl(QUrl(link))

    def send_to_me(self):
        msg = f"Hi Naalonh, nis \nDevice ID: {self.device_id}\nnh"
        link = f"https://t.me/Naalonh?text={requests.utils.quote(msg)}"
        QDesktopServices.openUrl(QUrl(link))

    def retry_license_check(self):
        print("[Auto Retry] Rechecking license...")

        def threaded_check():
            try:
                if os.path.exists(".env"):
                    os.remove(".env")
                if os.path.exists("license_pig.txt"):
                    os.remove("license_pig.txt")

                if check_license():
                    QTimer.singleShot(0, self.on_license_success)
            except Exception as e:
                print(f"[Retry Error] {e}")

        threading.Thread(target=threaded_check, daemon=True).start()

    def on_license_success(self):
        self.recheck_timer.stop()
        
        if hasattr(self, "spinner_timer") and self.spinner_timer.isActive():
            self.spinner_timer.stop()

        self.label.setText("✅ License Valid!")
        self.status.setFont(QFont("Segoe UI", 9))
        self.status.setText("Welcome Brother")
        self.result = True
        QTimer.singleShot(2000, self.accept)

    def animate_spinner(self):
        emoji = self.spinner_frames[self.spinner_index % len(self.spinner_frames)]
        dots = self.dot_cycle[self.dot_index % len(self.dot_cycle)]
        self.label.setText(f"{emoji} ចាំភ្លេទ Brother{dots}")
        self.spinner_index += 1
        self.dot_index += 1

    def update_progress_animation(self):
        if self.progress_value < 99:
            self.progress_value += 1
            self.progress.setValue(self.progress_value)
        else:
            self.progress_timer.stop()

    def run_check(self):
        self.label.setText("🔐 Checking License...")
        self.status.setFont(QFont("Segoe UI", 9))
        self.status.setText("Please wait while we validate your license.")

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_value = 0
        self.progress_timer.start(20)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #0d47a1;
                border-radius: 2px;
                background-color: #e3f2fd;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #1976d2;
                border-radius: 2px;
            }
        """)
        QApplication.processEvents()

        self.current_progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.increment_progress)
        self.timer.start(15)  # speed of progress animation

    def increment_progress(self):
        if self.current_progress < 100:
            self.current_progress += 1
            self.progress.setValue(self.current_progress)
        else:
            self.timer.stop()
            start_time = datetime.now()
            valid = check_license()
            elapsed = (datetime.now() - start_time).total_seconds()
            delay_remaining = max(0, 2.0 - elapsed)

            def finish_check():
                if valid:
                    self.label.setText("✅ License Valid!")
                    self.status.setText("Welcome Brother")
                    self.result = True
                    QTimer.singleShot(2000, self.accept)
                else:
                    self.device_id = get_device_fingerprint()
                    self.label.setText("❌ Invalid License")
                    self.status.setText(f"Your Device ID:\n{self.device_id}")
                    self.progress.setStyleSheet("QProgressBar::chunk { background-color: red; border-radius: 2px;}")
                    self.copy_id_button.setVisible(True)
                    self.send_me_button.setVisible(True)
                    self.send_channel_button.setVisible(True)
                    self.result = False

            QTimer.singleShot(int(delay_remaining * 1000), finish_check)


class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

# --- GUI APP ---
class Logger(io.StringIO):
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter

    def write(self, text):
        if text.strip():
            self.emitter.log_signal.emit(text.strip())

    def flush(self):
        pass

# def load_fonts_from_fons():
#     font_map = {}
#     font_folder = "fonts"
#     for file in os.listdir(font_folder):
#         if file.endswith(".ttf") or file.endswith(".otf"):
#             font_path = os.path.join(font_folder, file)
#             font_id = QFontDatabase.addApplicationFont(font_path)
#             if font_id != -1:
#                 family = QFontDatabase.applicationFontFamilies(font_id)[0]
#                 font_map[file] = family
#             #     print(f"✅ Loaded {file} as '{family}'")
#             # else:
#             #     print(f"❌ Failed to load: {file}")
#     return font_map

class ReplayBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bot_running = False
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        icon_path = resource_path("logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"[WARN] logo.png not found at: {icon_path}")
        # self.fonts = load_fonts_from_fons()

        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
        """)

        # Optional: Drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(Qt.black)
        self.main_widget.setGraphicsEffect(shadow)

        self.setCentralWidget(self.main_widget)

        dialog = LicenseCheckingDialog()
        dialog.exec_()

        if dialog.result is not True:
            sys.exit()

        self.setup_global_hotkey()

        self.setWindowTitle("Naalonh")
        self.setFixedSize(600, 600)
        self.move(20, 20)
        
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # ensure_resources_gui(self)

        title_bar = QWidget()
        title_bar.setFixedHeight(35)
        title_bar.setStyleSheet("""
            background-color: #;
            border-radius: 5px;

        """)
        
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar.setLayout(title_bar_layout)
                # Now insert it at the top of the main layout
        self.main_layout.insertWidget(0, title_bar)
        logo = QLabel()
        pixmap = QPixmap(resource_path("logo.png")).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setStyleSheet("margin-left: 6px;")
        title_bar_layout.addWidget(logo)

        # Add spacing between logo and title
        title_bar_layout.addSpacing(5)

        self.user, self.expiry = get_current_user_info()

        title_label = QLabel(f"{self.user}\t( {self.expiry} )")
        font = QFont("Segoe UI", 10, QFont.Bold)
        title_label.setFont(font)
        title_label.setStyleSheet("color: #00b4d8; letter-spacing: 1px;")
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()

        button_style = """
        QPushButton {
            background-color: transparent;
            border: none;
            padding: 2px;
        }
        QPushButton:hover {
            background-color: #ade8f4;
            border-radius: 6px;
            padding: 0px;
        }
        """
        min_button = QPushButton()
        min_button.setIcon(svg_to_icon(resource_path("icons/square-minus-solid-full.svg"), 20))
        min_button.setIconSize(QSize(20, 20))
        min_button.setFixedSize(30, 30)
        min_button.setToolTip("Minimize")
        min_button.setStyleSheet(button_style)
        min_button.clicked.connect(self.showMinimized)
        title_bar_layout.addWidget(min_button)

        max_button = QPushButton()
        max_button.setIcon(svg_to_icon(resource_path("icons/square-regular-full.svg"), 20))
        max_button.setIconSize(QSize(20, 20))
        max_button.setFixedSize(30, 30)
        max_button.setToolTip("Maximize")
        max_button.setStyleSheet(button_style)
        max_button.clicked.connect(self.toggle_maximize)
        title_bar_layout.addWidget(max_button)

        close_button = QPushButton()
        close_button.setIcon(svg_to_icon(resource_path("icons/square-xmark-solid-full.svg"), 20))
        close_button.setIconSize(QSize(20, 20))
        close_button.setFixedSize(30, 30)
        close_button.setToolTip("Close")
        close_button.setStyleSheet(button_style)
        close_button.clicked.connect(self.close)
        title_bar_layout.addWidget(close_button)
        
        # Top-right aligned Dark Mode switch
        theme_row = QHBoxLayout()
        theme_row.addStretch()
        self.theme_toggle = ToggleSwitch()
        self.theme_toggle.setToolTip("Dark Mode")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        theme_row.addWidget(self.theme_toggle)
        self.main_layout.addLayout(theme_row)

        # poppins = self.fonts.get(resource_path("Poppins-SemiBold.ttf"))

        self.title_label = QLabel(f"Naalonh.Tool - {VERSION.replace('v', 'V.')}")
        self.title_label.setFont(QFont("Segoe UI", 12))
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.main_layout.addWidget(self.title_label)


        # ➕ Add update button aligned to right below title
        top_right_layout = QHBoxLayout()
        top_right_layout.addStretch()

        self.update_button = QPushButton("Update")
        self.update_button.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.update_button.setFixedHeight(28)
        self.update_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
                color: #1976d2;
                border: 1px solid #1976d2;
            }
        """)
        self.update_button.clicked.connect(self.run_update_check)

        # Add Update button to layout
        self.main_layout.addWidget(self.update_button)

        # Create the ComboBox below the Update button
        self.combo_box = QComboBox()
        self.combo_box.setFixedHeight(25)  # Adjust size as needed
        self.combo_box.addItems(["Kill PIG", "Clear Mail", "Empty"])  # You can add more options
        self.combo_box.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.combo_box.setStyleSheet("color: #00b4d8; letter-spacing: 1px;")
        # Add ComboBox to layout
        self.main_layout.addWidget(self.combo_box)

        self.setCentralWidget(self.main_widget)
        
        top_right_layout.addWidget(self.update_button)
        self.main_layout.addLayout(top_right_layout)

        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        # Main status label
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 12))
        status_layout.addWidget(self.status_label)

        self.status_group.setLayout(status_layout)
        self.main_layout.addWidget(self.status_group)

        self.button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout()

        # ✅ 1. Add Progress Label at the top
        # Create labels
        label_total = QLabel("Total: 0")
        label_wins = QLabel("Wins: 0")
        label_deaths = QLabel("Deaths: 0")
        # match_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        # Set font if needed
        font = QFont(QFont("Segoe UI", 12, QFont.Bold))
        for label in (label_total, label_wins, label_deaths):
            label.setFont(font)

        # Layout
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(label_total)
        progress_layout.addStretch()           # pushes next widgets to the right
        progress_layout.addWidget(label_wins)
        progress_layout.addStretch()           # pushes final widget to the right end
        progress_layout.addWidget(label_deaths)

        # Container widget
        progress_container = QWidget()
        progress_container.setLayout(progress_layout)

        # Then add it to your main layout
        button_layout.addWidget(progress_container)

        # Optional: assign labels as instance attributes if you need to update them later
        self.label_total = label_total
        self.label_wins = label_wins
        self.label_deaths = label_deaths
        
        # --- MATCH ROW (Top)
        match_row = QHBoxLayout()
        match_row.setSpacing(6)

        match_label = QLabel("Match") 
        match_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        # match_label.setFont(QFont(poppins, 10, QFont.Bold))
        match_row.addWidget(match_label)

        self.repeat_input = QLineEdit()
        self.repeat_input.setPlaceholderText("Enter Match")
        self.repeat_input.setFixedWidth(120)
        self.repeat_input.setAlignment(Qt.AlignCenter)
        match_row.addWidget(self.repeat_input)

        match_row.addStretch()

        # Load saved match count
        if os.path.exists("Match.txt"):
            with open("Match.txt", "r") as f:
                saved_value = f.read().strip()
                if saved_value.isdigit():
                    self.repeat_input.setText(saved_value)
                    self.label_total.setText(f"Total: 0/{saved_value}")
                    self.label_wins.setText("Wins: 0")
                    self.label_deaths.setText("Deaths: 0")
                    
        # --- MAIL CLEAR ROW (Bottom)
        mail_row = QHBoxLayout()
        mail_row.setSpacing(4)

        self.clear_mail_checkbox = QCheckBox()
        self.clear_mail_checkbox.setFixedSize(18, 18)
        mail_row.addWidget(self.clear_mail_checkbox)


        clear_mail_label = QLabel("Clear Mail")
        clear_mail_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        # self.clear_mail_label.setFont(QFont("Segoe UI", 10))
        clear_mail_label.setStyleSheet("color: #1976d2;")
        mail_row.addWidget(clear_mail_label)

        self.clear_mail_interval_input = QLineEdit()
        self.clear_mail_interval_input.setPlaceholderText("25")
        self.clear_mail_interval_input.setFixedWidth(40)
        self.clear_mail_interval_input.setEnabled(False)
        self.clear_mail_interval_input.setStyleSheet("padding: 1px 3px;")
        self.clear_mail_interval_input.setValidator(QIntValidator(1, 25))
        mail_row.addWidget(self.clear_mail_interval_input)

        self.clear_mail_interval_input.editingFinished.connect(self.validate_mail_input)
        self.clear_mail_interval_input.textChanged.connect(self.enforce_max_mail_value)

        mail_row.addStretch()

        # Toggle enable/disable input
        self.clear_mail_checkbox.toggled.connect(
            lambda checked: self.clear_mail_interval_input.setEnabled(checked)
        )

        # Add both rows to button layout
        button_layout.addLayout(match_row)
        button_layout.addLayout(mail_row)
        button_row = QHBoxLayout()

        # --- MAT.exe Auto Restart Block
        mat_row = QHBoxLayout()

        self.restart_mat_checkbox = QCheckBox("Restart GAME")
        self.restart_mat_checkbox.setFont(QFont("Segoe UI", 10, QFont.Bold))
        mat_row.addWidget(self.restart_mat_checkbox)

        self.restart_interval_input = QLineEdit()
        self.restart_interval_input.setPlaceholderText("1")  # default 1 min
        self.restart_interval_input.setFixedWidth(40)
        self.restart_interval_input.setValidator(QIntValidator(1, 60))
        mat_row.addWidget(self.restart_interval_input)

        interval_label = QLabel("min")
        mat_row.addWidget(interval_label)
        mat_row.addStretch()

        button_layout.addLayout(mat_row)

        # Mode Selector
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        mode_row.addWidget(mode_label)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Crazy", "God"])
        self.mode_selector.setFixedWidth(120)
        mode_row.addWidget(self.mode_selector)
        mode_row.addStretch()
        button_layout.addLayout(mode_row)

        self.save_button = QPushButton("Save Match")
        self.save_button.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.save_button.setStyleSheet("""
        QPushButton
            {
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.save_button.clicked.connect(self.save_repeat_input)
        
        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Segoe UI", 12, QFont.Bold))

        self.start_button.setStyleSheet("""
        QPushButton
            { 
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.start_button.clicked.connect(self.toggle_bot)

        self.login_button = QPushButton("Save Pw Hero")
        self.login_button.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.login_button.setStyleSheet("""
        QPushButton
            { 
            background-color: #1976d2; 
            color: white;
            border: #1976d2;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover
            {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        """)
        self.login_button.clicked.connect(self.run_login_sequence)

        button_row.addWidget(self.save_button)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.login_button)

        button_layout.addLayout(button_row)


        self.button_group.setLayout(button_layout)
        self.main_layout.addWidget(self.button_group)

        # Custom Log Window Wrapper
        self.log_window = QWidget()
        self.log_window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.log_window.setAttribute(Qt.WA_TranslucentBackground)
        self.log_window.setWindowTitle("Activity Log")
        self.log_window.resize(420, 140)

        log_layout = QVBoxLayout(self.log_window)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)

        # Custom title bar
        log_title_bar = QWidget()
        log_title_bar.setFixedHeight(30)
        log_title_bar.setStyleSheet("background-color: #111111; border-top-left-radius: 6px; border-top-right-radius: 6px;")
        log_title_layout = QHBoxLayout(log_title_bar)
        log_title_layout.setContentsMargins(10, 0, 10, 0)

        log_label = QLabel("Close Tool - Ctrl+Alt+N")
        log_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        log_label.setStyleSheet("color: #ffffff; letter-spacing: 2px;")

        log_title_layout.addStretch()
        log_title_layout.addWidget(log_label)
        log_title_layout.addStretch()

        log_layout.addWidget(log_title_bar)

        # Log content
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        # self.log_output.setTextInteractionFlags(Qt.NoTextInteraction)
        self.log_output.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)  
        self.log_output.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas;
                font-size: 10pt;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 6px;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                width: 0px;
                height: 0px;
                background: transparent;
            }
        """)
        log_layout.addWidget(self.log_output)

        # Hotkeys
        QShortcut(QKeySequence("Ctrl+Alt+N"), self).activated.connect(self.close_app_by_hotkey)
        QShortcut(QKeySequence("Ctrl+Alt+N"), self.log_window).activated.connect(self.close_app_by_hotkey)


        # Show log
        screen_geometry = QDesktopWidget().availableGeometry()
        self.log_window.move(0, screen_geometry.height() - self.log_window.height())
        self.log_window.show()

        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.log_output.appendPlainText)

        sys.stdout = Logger(self.log_emitter)
        sys.stderr = Logger(self.log_emitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        QTimer.singleShot(50, lambda: self.apply_rounded_corners(radius=10))
    
    def mat_restart_loop(self):
        while self.bot_running and self.restart_mat_checkbox.isChecked():
            try:
                interval_text = self.restart_interval_input.text().strip()
                minutes = int(interval_text) if interval_text else 1
            except ValueError:
                minutes = 1

            print(f"[MAT] Waiting {minutes} minutes before restarting MAT.exe")
            time.sleep(minutes * 60)

            if not self.bot_running or not self.restart_mat_checkbox.isChecked():
                break

            os.system("taskkill /f /im MAT.exe")
            print("[MAT] Killed MAT.exe")

            stop_flag.set()  # ✅ Stop all ongoing replay logic
            print("[BOT] stop_flag set — replay will exit if running.")

            open_saved_app_login()
            print("[MAT] Relaunch command sent via app_path.txt")

            # ✅ Trigger login sequence again
            time.sleep(2)  # Wait a bit for the app to launch
            try:
                print("[MAT] Re-running login sequence...")
                LoginBotGUI(mode=self.mode_selector.currentText()).run_sequence_login()
            except Exception as e:
                print(f"[MAT ERROR] Failed to re-run login: {e}")

    def toggle_theme(self, state):
        if state == Qt.Checked:
            dark_stylesheet = """
                QWidget {
                    background-color: #2b2b2b;
                    color: #f0f0f0;
                }
                QPushButton {
                    background-color: #444;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QLineEdit, QPlainTextEdit {
                    background-color: #3c3c3c;
                    color: white;
                    border: 1px solid #555;
                }
                QGroupBox {
                    border: 1px solid #555;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 3px;
                }
                QLabel {
                    color: #dddddd;
                }
            """
            self.setStyleSheet(dark_stylesheet)
        else:
            self.setStyleSheet("")  # Reset to default

    def run_update_check(self):
        reply = QMessageBox.question(self, "Update", "Check for new version?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            check_for_updates()

    def setup_global_hotkey(self):
        import keyboard
        keyboard.add_hotkey("ctrl+alt+n", self.close_app_by_hotkey)

    def close_app_by_hotkey(self):
        print("[HOTKEY] Ctrl+Alt+N pressed — closing app.")

        try:
            # ✅ Stop all bot activity
            stop_flag.set()
            playback_flag.clear()

            # Optional: delay slightly to allow threads to finish
            time.sleep(0.3)

            # ✅ Close the log window if open
            if hasattr(self, "log_window") and self.log_window:
                self.log_window.close()
                self.log_window.deleteLater()
                print("[LOG] Log window closed.")

            # ✅ Force kill all hotkeys if using `keyboard` lib
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception as e:
            print(f"[ERROR] Closing via hotkey: {e}")

        QApplication.quit()

    def enforce_max_mail_value(self, text):
        try:
            val = int(text)
            if val > 25:
                self.clear_mail_interval_input.setText("25")
        except:
            pass  # Do nothing if they type non-numeric stuff

    def validate_mail_input(self):
        try:
            val = int(self.clear_mail_interval_input.text())
            if val > 25:
                self.clear_mail_interval_input.setText("25")
            elif val < 1:
                self.clear_mail_interval_input.setText("1")
        except:
            self.clear_mail_interval_input.setText("25")

    def apply_rounded_corners(self, radius=10):
        rect = self.rect()
        rounded = QRegion(rect, QRegion.Rectangle)
        path = QRegion(rect.adjusted(0, 0, 0, 0), QRegion.Rectangle)
        path = QRegion(rect.adjusted(0, 0, 0, 0), QRegion.Rectangle)
        corner = QRegion(0, 0, radius * 2, radius * 2, QRegion.Ellipse)

        rounded = QRegion(rect)
        rounded -= QRegion(0, 0, radius, radius)  # Top-left
        rounded -= QRegion(self.width() - radius, 0, radius, radius)  # Top-right
        rounded -= QRegion(0, self.height() - radius, radius, radius)  # Bottom-left
        rounded -= QRegion(self.width() - radius, self.height() - radius, radius, radius)  # Bottom-right

        rounded += QRegion(0, 0, radius * 2, radius * 2, QRegion.Ellipse)
        rounded += QRegion(self.width() - radius * 2, 0, radius * 2, radius * 2, QRegion.Ellipse)
        rounded += QRegion(0, self.height() - radius * 2, radius * 2, radius * 2, QRegion.Ellipse)
        rounded += QRegion(self.width() - radius * 2, self.height() - radius * 2, radius * 2, radius * 2, QRegion.Ellipse)

        self.setMask(rounded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_rounded_corners(radius=10)

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.drag_pos)
            self.drag_pos = event.globalPos()

    # Save  Matching
    def save_repeat_input(self):
        value = self.repeat_input.text().strip()
        if value:
            with open("Match.txt", "w") as f:
                f.write(value)
            
            # 🟢 Update progress label here
            self.progress_label.setText(f"Total: 0/{value} | Wins: 0 | Deaths: 0")

            QMessageBox.information(self, "Saved", f"Saved: {value} match(es)")
        else:
            QMessageBox.warning(self, "Empty", "Textbox is empty. Nothing saved.")

    # Function For enter matching
    def run_matching_game_multiple(self):
        print("[DEBUG] Entered run_matching_game_multiple")
        try:
            count = int(self.repeat_input.text().strip())
            if count <= 0:
                raise ValueError

            total_matches = count
            win_count = 0
            death_count = 0
            stop_flag.clear()

            # Show initial state before loop
            self.status_label.setText(f"Starting matches...")
            self.status_label.setStyleSheet("color: teal;")
            self.label_total.setText(f"Total: 0/{total_matches}")
            self.label_wins.setText("Wins: 0")
            self.label_deaths.setText("Deaths: 0")


            for i in range(1, total_matches + 1):
                death_flag.clear()

                # # ✅ FIX: clear stop flag early to allow next match
                # if stop_flag.is_set():
                #     if death_flag.is_set():
                #         print(f"[INFO] Continuing match {i} after death recovery.")
                #         stop_flag.clear()
                #     else:
                #         print(f"[INTERRUPT] Match stopped before run #{i} — exiting match loop.")
                #         break

                self.status_label.setText(f"Match {i}/{total_matches}")
                print(f"\nMatch {i} / {total_matches}")

                result, should_restart = self.run_single_match_with_result()

                if result == "win":
                    win_count += 1
                    time.sleep(8)
                elif result == "death":
                    time.sleep(3)
                    death_count += 1

                self.label_total.setText(f"Total: {win_count}/{total_matches}")
                self.label_wins.setText(f"Wins: {win_count}")
                self.label_deaths.setText(f"Deaths: {death_count}")

                if should_restart and death_flag.is_set():
                    print("[DEATH] Restarting game due to death.")
                    # restart_game_auto()
                    stop_flag.clear()  # ✅ allow match loop to continue



                # --- 💡 Check for alert/noise during the loop
                if human_like_click_image_auto("confirm1.png", confidence=0.85):
                    print(f"[ALERT] Noise alert triggered during match {i}. Clicking confirm1.png...")

                # --- 💡 NEW: Mail cleaner every 20 match or at last match
                if self.clear_mail_checkbox.isChecked():
                    try:
                        text = self.clear_mail_interval_input.text().strip()
                        interval = int(text) if text else 25
                        if interval > 25:
                            interval = 25
                            self.clear_mail_interval_input.setText("25")
                        elif interval < 1:
                            interval = 1
                            self.clear_mail_interval_input.setText("1")
                    except ValueError:
                        interval = 25
                        self.clear_mail_interval_input.setText("25")

                    if result == "win" and (
                        (win_count > 0 and win_count % interval == 0) or i == total_matches
                    ):
                        print(f"[INFO] Match {i} — Clearing mail (interval={interval})…")
                        stop_flag.set()
                        time.sleep(0.5)

                        def safe_mail_cleanup():
                            process_mail_sequence()

                        cleanup_thread = threading.Thread(target=safe_mail_cleanup)
                        cleanup_thread.start()
                        cleanup_thread.join()
                        stop_flag.clear()
                        print("[MAIL] Cleanup done.")

                time.sleep(1)

        except ValueError:
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Invalid Input", "Please enter a positive whole number."))
            return
        finally:
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet("color: black;")
            self.status_bar.showMessage("Idle")
            self.label_total.setText("Total: 0/0")
            self.label_wins.setText("Wins: 0")
            self.label_deaths.setText("Deaths: 0")
            self.start_button.setText("Start")
            self.bot_running = False  # ← this was likely missing

    def run_single_match_with_result(self):
        try:
            playback_flag.clear()
            stop_flag.clear()

            wait_and_click_auto("start.png", "Start Button")
            wait_and_click_auto("confirm.png", "Confirm Button")

            spawn = detect_spawn_auto()
            print(f"[DEBUG] Spawn detected: {spawn} (type: {type(spawn)})")  # Add this debug line
            
            # Add validation
            if not isinstance(spawn, int) or spawn < 1 or spawn > 10:
                print(f"[ERROR] Invalid spawn number: {spawn}")
                return "unknown", False
                
            if spawn:
                died = [False]

                def monitor_death():
                    # print("[DEBUG] Death monitoring started")
                    if detect_death_auto():
                        died[0] = True
                        # print("[DEBUG] Death detected")

                death_thread = threading.Thread(target=monitor_death, daemon=True)
                death_thread.start()

                replay_actions_for_spawn_auto(spawn)
                death_thread.join(timeout=2)

                print(f"[DEBUG] Death flag after join: {died[0]}")
                if died[0]:
                    print("[INFO] Death detected — restarting game, continuing next match.")
                    time.sleep(0.5)
                    
                    return "death", True
                else:
                    return "win", False

            return "unknown", False

        except Exception as e:
            print(f"[Error in run_single_match_with_result]: {e}")
            return "error", False

        except Exception as e:
            print(f"[Error in run_single_match_with_result]: {e}")
            return "error", False

    def run_login_sequence(self):
        print("[GUI] Launching Login Window...")
        self.login_window = LoginBotGUI()
        self.login_window.show()

    # def toggle_bot(self):
    #     input_text = self.repeat_input.text().strip()

    #     if not self.bot_running:
    #         stop_flag.clear()
    #         self.status_label.setText("Running")
    #         self.status_label.setStyleSheet("color: green;")
    #         self.status_bar.showMessage("ReplayBot started")
    #         print("[GUI] Triggering Login and ReplayBot Start")

    #         selected_mode = self.mode_selector.currentText()
    #         login_gui = LoginBotGUI(mode=selected_mode)
    #         login_gui.run_sequence_login()

    #         self.bot_running = True  # ✅ MUST COME FIRST

    #         if self.restart_mat_checkbox.isChecked():
    #             threading.Thread(target=self.mat_restart_loop, daemon=True).start()

    #         if input_text == "1":
    #             print("Running only 1 match")
    #             threading.Thread(target=run_single_match_auto, daemon=True).start()
    #         else:
    #             print("Starting full auto-loop mode")
    #             threading.Thread(target=self.run_matching_game_multiple, daemon=True).start()
            
    #         self.start_button.setText("Stop")

    #     else:
    #         stop_flag.set()
    #         self.status_label.setText("Stopped")
    #         self.status_label.setStyleSheet("color: red;")
    #         self.status_bar.showMessage("ReplayBot stopped")
    #         print("[GUI] Triggering ReplayBot Stop")

    #         stop_replay_bot_auto()
    #         self.bot_running = False
    #         self.start_button.setText("Start")

    def toggle_bot(self):
        input_text = self.repeat_input.text().strip()
        selected_option = self.combo_box.currentText()

        if not self.bot_running:
            stop_flag.clear()
            self.status_label.setText("Running")
            self.status_label.setStyleSheet("color: green;")
            self.status_bar.showMessage("ReplayBot started")
            self.bot_running = True
            self.start_button.setText("Stop")

            # 🔁 Clear Mail ONLY → MAIL SEQUENCE
            if selected_option == "Clear Mail":
                print("[MODE] Running mail processing only (Clear Mail selected)")

                def run_mail_then_reset():
                    process_mail_sequence(skip_start_check=True)
                    # ✅ Reset bot state after mail finishes
                    self.status_label.setText("Idle")
                    self.status_label.setStyleSheet("color: black;")
                    self.status_bar.showMessage("Idle")
                    self.bot_running = False
                    self.start_button.setText("Start")

                threading.Thread(target=run_mail_then_reset, daemon=True).start()
                return

            # 🧠 Default behavior (Kill PIG, Option 3, etc.)
            selected_mode = self.mode_selector.currentText()
            login_gui = LoginBotGUI(mode=selected_mode)
            login_gui.run_sequence_login()

            if self.restart_mat_checkbox.isChecked():
                threading.Thread(target=self.mat_restart_loop, daemon=True).start()

            if input_text == "1":
                threading.Thread(target=run_single_match_auto, daemon=True).start()
            else:
                threading.Thread(target=self.run_matching_game_multiple, daemon=True).start()

        else:
            stop_flag.set()
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("color: red;")
            self.status_bar.showMessage("ReplayBot stopped")
            stop_replay_bot_auto()
            self.bot_running = False
            self.start_button.setText("Start")
            
    def closeEvent(self, event):
        try:
            stop_flag.set()
            playback_flag.clear()

            if os.path.exists(".env"):
                os.remove(".env")
                print("[LOG] Deleted .env in closeEvent.")

            if hasattr(self, "log_window") and self.log_window:
                self.log_window.close()
                self.log_window.deleteLater()

            import keyboard
            keyboard.unhook_all_hotkeys()

            print("[LOG] Clean exit triggered.")
        except Exception as e:
            print(f"[ERROR] closeEvent: {e}")
        event.accept()

atexit.register(lambda: os.kill(os.getpid(), signal.SIGTERM))

if __name__ == "__main__":
    check_for_updates()
    
    # ⬇ Add this to activate F3 toggle logic
    from pynput import keyboard
    keyboard.Listener(on_press=on_press_auto).start()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(resource_path("logo.png")))
    window = ReplayBotGUI()
    window.show()
    sys.exit(app.exec_())



