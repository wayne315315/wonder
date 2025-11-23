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
            raise e  # <--- CRITICAL: Re-raise error so caller knows connection failed

    def disconnect(self):
        if self.ftp:
            try:
                self.ftp.quit()
            except:
                try:
                    self.ftp.close()
                except:
                    pass
            print("🔌 Disconnected.")

    # ---------------------------------------------------------
    # HELPER: Check remote type
    # ---------------------------------------------------------
    def _is_remote_dir(self, path):
        """Checks if a remote path is a directory using MLSD."""
        if path in [".", "/", ""]:
            return True
            
        parent = os.path.dirname(path)
        if parent == "": parent = "."
        target_name = os.path.basename(path)
        
        try:
            for name, facts in self.ftp.mlsd(parent):
                if name == target_name:
                    return facts.get('type') == 'dir'
        except error_perm:
            # If we can't check it, we assume it's not a dir or doesn't exist.
            # The subsequent download attempt will fail and raise the correct error.
            return False 
        return False

    # ---------------------------------------------------------
    # LISTING
    # ---------------------------------------------------------
    def list_files(self, directory=".", recursive=False):
        print(f"📂 Listing '{directory}'...")
        def _walk(current_path, level):
            try:
                items = list(self.ftp.mlsd(current_path))
                for name, facts in items:
                    if name in ['.', '..']: continue
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
                # We usually don't raise here because listing failure shouldn't crash the app
                pass 

        _walk(directory, 0)

    # ---------------------------------------------------------
    # DIRECTORY MANAGEMENT
    # ---------------------------------------------------------
    def mkdir(self, directory):
        parts = directory.replace("\\", "/").split("/")
        path = ""
        for part in parts:
            if not part: continue
            path = f"{path}/{part}" if path else part
            try:
                self.ftp.mkd(path)
                print(f"   ✨ Created: {path}")
            except error_perm:
                pass

    # ---------------------------------------------------------
    # UPLOAD (Recursive)
    # ---------------------------------------------------------
    def upload(self, local_path, remote_path=None):
        if not remote_path:
            remote_path = local_path

        if os.path.isdir(local_path):
            self._upload_dir(local_path, remote_path)
        elif os.path.isfile(local_path):
            self.mkdir(os.path.dirname(remote_path))
            self._upload_file(local_path, remote_path)
        else:
            msg = f"❌ Local path not found: {local_path}"
            print(msg)
            # Raise error so the 'try/except' block catches it
            raise FileNotFoundError(msg)

    def _upload_file(self, local_path, remote_path):
        print(f"⬆️ Uploading file: {remote_path}...")
        try:
            with open(local_path, "rb") as f:
                self.ftp.storbinary(f"STOR {remote_path}", f)
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")
            raise e # <--- CRITICAL ADDITION

    def _upload_dir(self, local_dir, remote_dir):
        print(f"📦 Uploading folder: {local_dir} -> {remote_dir}")
        self.mkdir(remote_dir)
        for root, dirs, files in os.walk(local_dir):
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == ".": rel_path = ""
            current_remote_dir = f"{remote_dir}/{rel_path}" if rel_path else remote_dir
            
            for d in dirs:
                self.mkdir(f"{current_remote_dir}/{d}")
            for f in files:
                local_f = os.path.join(root, f)
                remote_f = f"{current_remote_dir}/{f}"
                self._upload_file(local_f, remote_f)
        print("✅ Folder upload complete.")

    # ---------------------------------------------------------
    # DOWNLOAD (Recursive)
    # ---------------------------------------------------------
    def download(self, remote_path, local_path=None):
        if not local_path:
            local_path = remote_path

        # If remote_path doesn't exist, _is_remote_dir returns False.
        # It then goes to _download_file, which tries to RETR.
        # RETR fails, raises Exception, and your main script catches it.
        if self._is_remote_dir(remote_path):
            self._download_dir(remote_path, local_path)
        else:
            self._download_file(remote_path, local_path)

    def _download_file(self, remote_path, local_path):
        print(f"⬇️ Downloading file: {remote_path}...")
        try:
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            with open(local_path, "wb") as f:
                self.ftp.retrbinary(f"RETR {remote_path}", f.write)
        except Exception as e:
            # Cleanup partial file
            if os.path.exists(local_path):
                os.remove(local_path) 
            print(f"❌ Failed to download {remote_path}: {e}")
            raise e # <--- CRITICAL: Re-raise so the main script knows it failed!

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
            raise e # <--- CRITICAL ADDITION

    # ---------------------------------------------------------
    # DELETE (Recursive rm -rf)
    # ---------------------------------------------------------
    def delete(self, remote_path):
        if self._is_remote_dir(remote_path):
            self._delete_dir_recursive(remote_path)
        else:
            print(f"🗑️ Deleting file: {remote_path}")
            try:
                self.ftp.delete(remote_path)
            except Exception as e:
                print(f"❌ Delete failed: {e}")
                raise e # <--- Added raise

    def _delete_dir_recursive(self, remote_dir):
        print(f"💥 Recursively deleting folder: {remote_dir}")
        try:
            items = list(self.ftp.mlsd(remote_dir))
            for name, facts in items:
                if name in ['.', '..']: continue
                full_path = f"{remote_dir}/{name}"
                if facts.get('type') == 'dir':
                    self._delete_dir_recursive(full_path)
                else:
                    self.ftp.delete(full_path)
            self.ftp.rmd(remote_dir)
        except Exception as e:
            print(f"❌ Error deleting folder {remote_dir}: {e}")
            raise e # <--- Added raise

    # UTILS
    def rename(self, old, new):
        print(f"✏️ Renaming {old} -> {new}")
        try:
            self.ftp.rename(old, new)
        except Exception as e:
            print(f"❌ Rename failed: {e}")
            raise e
