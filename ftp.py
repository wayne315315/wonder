import os
import sys
from ftplib import FTP, error_perm

class RayFTPClient:
    """
    A wrapper around ftplib to easily manage files on the Desktop Worker.
    """
    def __init__(self, host="100.111.210.199", port=2121, user="rayuser", passwd="raypass"):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.ftp = None

    def __enter__(self):
        """Allows usage with 'with' statement."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically closes connection."""
        self.disconnect()

    def connect(self):
        """Establishes connection to the Desktop."""
        try:
            self.ftp = FTP()
            print(f"🌐 Connecting to {self.host}:{self.port}...")
            self.ftp.connect(self.host, self.port)
            self.ftp.login(self.user, self.passwd)
            
            # CRITICAL: Passive mode is required for VPN/NAT traversal
            self.ftp.set_pasv(True) 
            print(f"✅ Connected to Desktop. ({self.ftp.getwelcome()})")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            raise e

    def disconnect(self):
        """Closes the connection gracefully."""
        if self.ftp:
            try:
                self.ftp.quit()
                print("🔌 Disconnected.")
            except:
                self.ftp.close()

    # ---------------------------------------------------------
    # CORE OPERATIONS
    # ---------------------------------------------------------

    def list_files(self, directory="."):
        """Lists files in the specified remote directory."""
        print(f"📂 Listing directory: '{directory}'")
        try:
            files = []
            self.ftp.cwd(directory)
            self.ftp.retrlines('LIST', files.append)
            for f in files:
                print(f"   - {f}")
            return files
        except Exception as e:
            print(f"❌ List failed: {e}")
            return []

    def upload(self, local_path, remote_name=None):
        """Uploads a local file to the Desktop."""
        if not remote_name:
            remote_name = os.path.basename(local_path)

        if not os.path.exists(local_path):
            print(f"❌ Error: Local file not found: {local_path}")
            return

        print(f"⬆️ Uploading '{local_path}' -> '{remote_name}'...")
        try:
            with open(local_path, "rb") as f:
                self.ftp.storbinary(f"STOR {remote_name}", f)
            print("✅ Upload success.")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

    def download(self, remote_name, local_path=None):
        """Downloads a file from the Desktop to Mac."""
        if not local_path:
            local_path = remote_name

        print(f"⬇️ Downloading '{remote_name}' -> '{local_path}'...")
        try:
            with open(local_path, "wb") as f:
                self.ftp.retrbinary(f"RETR {remote_name}", f.write)
            print("✅ Download success.")
        except Exception as e:
            print(f"❌ Download failed (File might not exist): {e}")
            # Clean up empty file if download failed
            if os.path.exists(local_path):
                os.remove(local_path)

    def rename(self, old_name, new_name):
        """Renames (or moves) a file on the Desktop."""
        print(f"✏️ Renaming '{old_name}' to '{new_name}'...")
        try:
            self.ftp.rename(old_name, new_name)
            print("✅ Rename success.")
        except Exception as e:
            print(f"❌ Rename failed: {e}")

    def move(self, filename, destination_folder):
        """Moves a file into a different folder on the Desktop."""
        # Ensure destination ends with filename
        new_path = f"{destination_folder}/{filename}"
        self.rename(filename, new_path)

    def delete_file(self, filename):
        """Deletes a specific file on the Desktop."""
        print(f"🗑️ Deleting file '{filename}'...")
        try:
            self.ftp.delete(filename)
            print("✅ File deleted.")
        except Exception as e:
            print(f"❌ Delete failed: {e}")

    def mkdir(self, directory):
        """Creates a new directory on the Desktop."""
        print(f"fo Create directory '{directory}'...")
        try:
            self.ftp.mkd(directory)
            print("✅ Directory created.")
        except Exception as e:
            print(f"❌ Mkdir failed (Might already exist): {e}")

    def rmdir(self, directory):
        """Removes an EMPTY directory on the Desktop."""
        print(f"🗑️ Removing directory '{directory}'...")
        try:
            self.ftp.rmd(directory)
            print("✅ Directory removed.")
        except Exception as e:
            print(f"❌ Rmdir failed (Directory must be empty): {e}")

# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Using 'with' automatically handles connect/disconnect
    with RayFTPClient() as client:
        # 1. List current files
        client.list_files()

        # 2. Upload a test file
        with open("test_upload.txt", "w") as f:
            f.write("Hello Desktop!")
        client.upload("test_upload.txt", "uploaded_test.txt")

        # 3. Rename it
        client.rename("uploaded_test.txt", "renamed_test.txt")

        # 4. Create a folder
        client.mkdir("models_archive")

        # 5. Move file into folder
        # Note: pyftpdlib supports standard unix-like paths even on Windows
        client.rename("renamed_test.txt", "models_archive/renamed_test.txt")

        # 6. Download it back
        client.download("models_archive/renamed_test.txt", "downloaded_back.txt")

        # 7. Cleanup (Delete file and folder)
        client.delete_file("models_archive/renamed_test.txt")
        client.rmdir("models_archive")
        
        print("\n✨ All operations completed successfully.")