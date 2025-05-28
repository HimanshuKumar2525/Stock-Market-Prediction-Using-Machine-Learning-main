import sqlite3
import hashlib
from datetime import datetime

# =======================
# 📦 Initialize DB connection
# =======================
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# =======================
# 📌 Create tables if not exist
# =======================
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS register_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    registered_at TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    login_time TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    prediction_result TEXT,
    predicted_at TEXT)''')

conn.commit()

# =======================
# 🔒 Hash password using SHA256
# =======================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =======================
# 👤 Add new user with registration log
# =======================
def add_user(username, password):
    try:
        hashed_pw = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
        conn.commit()
        c.execute('INSERT INTO register_logs (username, registered_at) VALUES (?, ?)',
                  (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print(f"✅ User '{username}' registered successfully.")
    except sqlite3.IntegrityError:
        print(f"⚠️  Username '{username}' already exists.")

# =======================
# 🔐 Authenticate user and log login time
# =======================
def login_user(username, password):
    hashed_pw = hash_password(password)
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_pw))
    user = c.fetchone()
    if user:
        c.execute('INSERT INTO login_logs (username, login_time) VALUES (?, ?)',
                  (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print(f"✅ Welcome, {username}! Logged in successfully.")
        return True
    else:
        print("❌ Invalid username or password.")
        return False

# =======================
# 📊 Store prediction result for user
# =======================
def store_prediction(username, result):
    c.execute('INSERT INTO predictions (username, prediction_result, predicted_at) VALUES (?, ?, ?)',
              (username, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    print(f"✅ Prediction logged for {username}.")

# =======================
# 📋 Add predefined default user if not exists
# =======================
def add_default_user():
    username = "Him8850"
    password = "Him@8850"
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    if not c.fetchone():
        add_user(username, password)
        print(f"Default user '{username}' added.")
    else:
        print(f"User '{username}' already exists.")

# =======================
# 📖 Display all logs for admin view
# =======================
def show_all_logs():
    print("\n--- 📋 Users ---")
    for row in c.execute('SELECT * FROM users'):
        print(row)
    print("\n--- 📖 Register Logs ---")
    for row in c.execute('SELECT * FROM register_logs'):
        print(row)
    print("\n--- 🔐 Login Logs ---")
    for row in c.execute('SELECT * FROM login_logs'):
        print(row)
    print("\n--- 📊 Predictions ---")
    for row in c.execute('SELECT * FROM predictions'):
        print(row)

# =======================
# 🎛️ CLI Menu System
# =======================
def main():
    add_default_user()

    while True:
        print("\n========== 📌 MENU 📌 ==========")
        print("1. Register New User")
        print("2. Login")
        print("3. Show All Logs")
        print("4. Exit")
        choice = input("Select option (1-4): ")

        if choice == '1':
            username = input("Enter new username: ")
            password = input("Enter new password: ")
            add_user(username, password)

        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")
            if login_user(username, password):
                while True:
                    print("\n✅ Prediction Menu")
                    print("1. Store Prediction")
                    print("2. Logout")
                    sub_choice = input("Select option (1-2): ")
                    if sub_choice == '1':
                        result = input("Enter prediction result: ")
                        store_prediction(username, result)
                    elif sub_choice == '2':
                        print("🔓 Logged out.")
                        break
                    else:
                        print("⚠️  Invalid choice.")

        elif choice == '3':
            show_all_logs()

        elif choice == '4':
            print("👋 Exiting program. Goodbye!")
            break

        else:
            print("⚠️  Invalid option, please try again.")

# =======================
# ▶️ Run CLI App
# =======================
if __name__ == "__main__":
    main()
