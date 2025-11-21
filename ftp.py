import os
import sys
import shutil
import time
from ftplib import FTP, error_perm

class RayFTPClient:
    """
    A robust FTP client that handles recursive uploads, downloads, and deletions.
    Designed for Ray clusters connecting via VPN/Meshnet.
    """
    def __init__(self, host="100.111.210.199", port=2121, user="rayuser", passwd="raypass"):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.ftp = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        try:
            self.ftp = FTP()
            print(f"🌐 Connecting to {self.host}:{self.port}...")
            self.ftp.connect(self.host, self.port)
            self.ftp.login(self.user, self.passwd)
            self.ftp.set_pasv(True) # Critical for VPN
            print(f"✅ Connected to Desktop. ({self.ftp.getwelcome()})")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            raise e

    def disconnect(self):
        if self.ftp:
            try:
                self.ftp.quit()
            except:
                self.ftp.close()
            print("🔌 Disconnected.")

    # ---------------------------------------------------------
    # HELPER: Check remote type
    # ---------------------------------------------------------
    def _is_remote_dir(self, path):
        """Checks if a remote path is a directory using MLSD."""
        # Root is always a dir
        if path == "." or path == "/" or path == "":
            return True
            
        parent = os.path.dirname(path)
        if parent == "": parent = "."
        target_name = os.path.basename(path)
        
        try:
            # List parent to find target's metadata
            for name, facts in self.ftp.mlsd(parent):
                if name == target_name:
                    return facts.get('type') == 'dir'
        except error_perm:
            return False # Likely doesn't exist or no permission
        return False

    # ---------------------------------------------------------
    # LISTING
    # ---------------------------------------------------------
    def list_files(self, directory=".", recursive=False):
        """Lists files/folders. If recursive=True, walks the tree."""
        print(f"📂 Listing '{directory}'...")
        
        def _walk(current_path, level):
            try:
                items = list(self.ftp.mlsd(current_path))
                for name, facts in items:
                    if name in ['.', '..']: continue
                    
                    # Build clean path
                    full_path = f"{current_path}/{name}" if current_path != "." else name
                    indent = "   " * level
                    
                    if facts.get('type') == 'dir':
                        print(f"{indent}📁 {name}/")
                        if recursive:
                            _walk(full_path, level + 1)
                    else:
                        print(f"{indent}📄 {name}")
            except error_perm as e:
                print(f"   ❌ Access Denied: {current_path}")

        _walk(directory, 0)

    # ---------------------------------------------------------
    # DIRECTORY MANAGEMENT
    # ---------------------------------------------------------
    def mkdir(self, directory):
        """Recursively creates directories (like mkdir -p)."""
        parts = directory.replace("\\", "/").split("/")
        path = ""
        for part in parts:
            if not part: continue
            path = f"{path}/{part}" if path else part
            try:
                self.ftp.mkd(path)
                print(f"   ✨ Created: {path}")
            except error_perm:
                # Ignore error if it already exists
                pass

    # ---------------------------------------------------------
    # UPLOAD (Recursive)
    # ---------------------------------------------------------
    def upload(self, local_path, remote_path=None):
        """
        Uploads a file OR a folder recursively.
        """
        if not remote_path:
            remote_path = local_path

        if os.path.isdir(local_path):
            self._upload_dir(local_path, remote_path)
        elif os.path.isfile(local_path):
            self.mkdir(os.path.dirname(remote_path))
            self._upload_file(local_path, remote_path)
        else:
            print(f"❌ Local path not found: {local_path}")

    def _upload_file(self, local_path, remote_path):
        print(f"⬆️ Uploading file: {remote_path}...")
        try:
            with open(local_path, "rb") as f:
                self.ftp.storbinary(f"STOR {remote_path}", f)
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")

    def _upload_dir(self, local_dir, remote_dir):
        print(f"📦 Uploading folder: {local_dir} -> {remote_dir}")
        # 1. Create the remote root folder
        self.mkdir(remote_dir)

        # 2. Walk local structure
        for root, dirs, files in os.walk(local_dir):
            # Calculate relative path to mirror structure
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == ".": rel_path = ""
            
            current_remote_dir = f"{remote_dir}/{rel_path}" if rel_path else remote_dir
            
            # Create subdirectories
            for d in dirs:
                self.mkdir(f"{current_remote_dir}/{d}")

            # Upload files
            for f in files:
                local_f = os.path.join(root, f)
                remote_f = f"{current_remote_dir}/{f}"
                self._upload_file(local_f, remote_f)
        print("✅ Folder upload complete.")

    # ---------------------------------------------------------
    # DOWNLOAD (Recursive)
    # ---------------------------------------------------------
    def download(self, remote_path, local_path=None):
        """
        Downloads a file OR a folder recursively.
        """
        if not local_path:
            local_path = remote_path

        if self._is_remote_dir(remote_path):
            self._download_dir(remote_path, local_path)
        else:
            self._download_file(remote_path, local_path)

    def _download_file(self, remote_path, local_path):
        print(f"⬇️ Downloading file: {remote_path}...")
        try:
            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            with open(local_path, "wb") as f:
                self.ftp.retrbinary(f"RETR {remote_path}", f.write)
        except Exception as e:
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"❌ Failed to download {remote_path}: {e}")

    def _download_dir(self, remote_dir, local_dir):
        print(f"📦 Downloading folder: {remote_dir} -> {local_dir}")
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            items = list(self.ftp.mlsd(remote_dir))
            for name, facts in items:
                if name in ['.', '..']: continue
                
                r_path = f"{remote_dir}/{name}"
                l_path = os.path.join(local_dir, name)

                if facts.get('type') == 'dir':
                    self._download_dir(r_path, l_path)
                else:
                    self._download_file(r_path, l_path)
        except Exception as e:
            print(f"❌ Error reading remote folder {remote_dir}: {e}")

    # ---------------------------------------------------------
    # DELETE (Recursive rm -rf)
    # ---------------------------------------------------------
    def delete(self, remote_path):
        """Deletes a file or folder recursively."""
        if self._is_remote_dir(remote_path):
            self._delete_dir_recursive(remote_path)
        else:
            print(f"🗑️ Deleting file: {remote_path}")
            try:
                self.ftp.delete(remote_path)
            except Exception as e:
                print(f"❌ Delete failed: {e}")

    def _delete_dir_recursive(self, remote_dir):
        print(f"💥 Recursively deleting folder: {remote_dir}")
        try:
            # Must delete contents first (Depth-First)
            items = list(self.ftp.mlsd(remote_dir))
            for name, facts in items:
                if name in ['.', '..']: continue
                
                full_path = f"{remote_dir}/{name}"
                if facts.get('type') == 'dir':
                    self._delete_dir_recursive(full_path)
                else:
                    self.ftp.delete(full_path)
            
            # Now directory is empty, remove it
            self.ftp.rmd(remote_dir)
        except Exception as e:
            print(f"❌ Error deleting folder {remote_dir}: {e}")

    # ---------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------
    def rename(self, old, new):
        print(f"✏️ Renaming {old} -> {new}")
        try:
            self.ftp.rename(old, new)
        except Exception as e:
            print(f"❌ Rename failed: {e}")

    def move(self, src, dest_folder):
        filename = os.path.basename(src)
        # Ensure no trailing slash issues
        if dest_folder.endswith("/"):
            dest_folder = dest_folder[:-1]
        new_path = f"{dest_folder}/{filename}"
        self.rename(src, new_path)

# ---------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure your 'ftp_manager.py' is running on the desktop first!
    
    with RayFTPClient(host="100.111.210.199") as client:
        
        # 1. Test recursive upload
        print("\n--- 1. Creating local test folder ---")
        os.makedirs("test_data_local/subfolder", exist_ok=True)
        with open("test_data_local/file1.txt", "w") as f: f.write("File 1")
        with open("test_data_local/subfolder/file2.txt", "w") as f: f.write("File 2")
        
        print("\n--- 2. Uploading Folder ---")
        client.upload("test_data_local", "remote_data_folder")
        print("\n--- 3. Listing Recursive ---")
        client.list_files("remote_data_folder", recursive=True)
        time.sleep(10)

        print("\n--- 4. Downloading Folder ---")
        client.download("remote_data_folder", "downloaded_data_folder")

        print("\n--- 5. Deleting Recursive ---")
        client.delete("remote_data_folder")
        
        # Verify deletion
        client.list_files()
        
        # Cleanup local
        import shutil
        if os.path.exists("test_data_local"): shutil.rmtree("test_data_local")
        if os.path.exists("downloaded_data_folder"): shutil.rmtree("downloaded_data_folder")
