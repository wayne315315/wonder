import ray
import os
import time
import threading
import sys

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
FTP_PORT = 2121
FTP_USER = "rayuser"
FTP_PASS = "raypass"
FTP_ROOT_DIR = "/home/linhsuanyu/ftp/"
PASSIVE_PORTS = range(60000, 60010)
ACTOR_NAME = "Global_Desktop_FTP"

# ---------------------------------------------------------
# THE PERSISTENT ACTOR (Runs on Desktop)
# ---------------------------------------------------------
@ray.remote(resources={"arch_x86": 0.001})
class PersistentFTPServer:
    def __init__(self):
        self.server = None
        self.thread = None
        self.is_running = False

    def start(self):
        """Starts the FTP server in a background thread."""
        if self.is_running:
            return "⚠️ FTP Server is ALREADY running."

        from pyftpdlib.authorizers import DummyAuthorizer
        from pyftpdlib.handlers import FTPHandler
        from pyftpdlib.servers import FTPServer

        # create root folder if not exists
        if not os.path.exists(FTP_ROOT_DIR):
            os.makedirs(FTP_ROOT_DIR, exist_ok=True)

        # Setup Auth and Directory
        authorizer = DummyAuthorizer()
        authorizer.add_user(FTP_USER, FTP_PASS, FTP_ROOT_DIR, perm="elradfmw")

        handler = FTPHandler
        handler.authorizer = authorizer
        handler.passive_ports = PASSIVE_PORTS
        
        # Start listening
        address = ("0.0.0.0", FTP_PORT)
        self.server = FTPServer(address, handler)
        self.server.max_cons = 10

        # Launch in thread so this Actor stays responsive to 'stop' calls
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.is_running = True
        
        return f"✅ FTP Server STARTED on Port {FTP_PORT} (PID: {os.getpid()})"

    def stop(self):
        """Stops the server and kills the thread."""
        if not self.is_running:
            return "⚠️ Server was not running."
        
        self.server.close_all()
        self.is_running = False
        return "🛑 FTP Server STOPPED."

    def get_status(self):
        return "🟢 Running" if self.is_running else "🔴 Stopped"

# ---------------------------------------------------------
# CLI LOGIC
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ftp.py [start|stop|status]")
        return

    command = sys.argv[1].lower()
    
    # Connect to Cluster
    ray.init(address='auto', namespace="ftp_service")

    if command == "start":
        try:
            # 1. Create DETACHED actor (lifetime="detached")
            # This ensures it survives even after this script exits!
            ftp_actor = PersistentFTPServer.options(
                name=ACTOR_NAME, 
                lifetime="detached",
                get_if_exists=True
            ).remote()
            
            print(ray.get(ftp_actor.start.remote()))
            print(f"📌 Actor '{ACTOR_NAME}' is now running in the background.")
            
        except Exception as e:
            print(f"❌ Failed to start: {e}")

    elif command == "stop":
        try:
            # 1. Connect to the EXISTING actor by name
            ftp_actor = ray.get_actor(ACTOR_NAME)
            
            # 2. Tell it to stop the FTP loop
            print(ray.get(ftp_actor.stop.remote()))
            
            # 3. Kill the actor process entirely
            ray.kill(ftp_actor)
            print(f"🗑️ Detached Actor '{ACTOR_NAME}' removed from cluster.")
            
        except ValueError:
            print(f"⚠️ Could not find actor '{ACTOR_NAME}'. Is it running?")

    elif command == "status":
        try:
            ftp_actor = ray.get_actor(ACTOR_NAME)
            print(f"Status: {ray.get(ftp_actor.get_status.remote())}")
        except ValueError:
            print("Status: 🔴 No Actor Found (Server is down)")

if __name__ == "__main__":
    main()