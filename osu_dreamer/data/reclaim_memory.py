
import os

# check if using WSL
if os.system("uname -r | grep microsoft > /dev/null") == 0:
    def reclaim_memory():
        """
        free the vm page cache - see `https://devblogs.microsoft.com/commandline/memory-reclaim-in-the-windows-subsystem-for-linux-2/`
        
        add to /etc/sudoers:
        %sudo ALL=(ALL) NOPASSWD: /bin/tee /proc/sys/vm/drop_caches
        """
        os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
else:
    def reclaim_memory():
        pass