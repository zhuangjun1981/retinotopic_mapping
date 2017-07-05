################################################################################
# Copyright 2015, The Allen Institute for Brain Science
# Author: Jed Perkins
################################################################################
import threading
import sys
from Queue import Queue, Empty

class _RThread(threading.Thread):
    """
    Private helper class for reading output from a file-like object
    in a thread so as not to block the main thread.

    Args:
        f : file-like object
            File to read from.
        q : Queue.Queue-like object
            Queue to put file output.
        name : string
            Name of the thread. Auto-generated if not supplied.

    """
    def __init__(self, f=None, q=None, name=None):
        """
        Constructor.

        """
        super(_RThread, self).__init__(target=self._read, name=name)
        self.daemon = True
        if f is None:
            self.f = sys.stdout
        else:
            self.f = f
        if q is None:
            raise AttributeError("q must be a Queue-like object")
        elif (not hasattr(q, 'put')) or (not hasattr(q, 'get_nowait')):
            raise AttributeError("q must be a Queue-like object")
        self.q = q

    def _read(self):
        """
        Read from the file and put line in Queue.

        """
        while True:
            self.q.put(self.f.readline())

class Reader(object):
    """
    Non-blocking file reader. Uses threading to keep the block out
    of the main thread.

    Args:
        f : file-like object
            File to read.

    """
    def __init__(self, f=None):
        """
        Constructor.

        """
        super(Reader, self).__init__()
        self.f = f
        self.q = Queue()
        self.thread = _RThread(f=self.f, q=self.q)
        self.thread.start()

    def read(self):
        """
        Read from the file. Return an empty string if nothing to read.

        Returns:
            line : string
                Line read from the file.

        """
        try:
            line = self.q.get_nowait()
        except Empty:
            line = ''
        return line

class SubprocessMonitor(object):
    """
    Monitor the stdout and/or stderr of a subprocess with assignable
    handlers. To monitor either, they must have been opened as
    subprocess.PIPE.

    Args:
        proc : process
            Process to monitor.
        stdout_handler : callable(str)
            Handler for lines from proc.stdout, defaults to print.
        stderr_handler : callable(str)
            Handler for lines from proc.stderr, defaults to print.

    """
    def __init__(self, proc, stdout_handler=None, stderr_handler=None):
        """
        Constructor.

        """
        super(SubprocessMonitor, self).__init__()
        self.proc = proc
        self.stdout_reader = None
        self.stderr_reader = None
        # Set handler for stdout
        if stdout_handler is None:
            self.stdout_handler = self._stdout_handler
        else:
            self.stdout_handler = stdout_handler
        # Set handler for stderr
        if stderr_handler is None:
            self.stderr_handler = self._stderr_handler
        else:
            self.stderr_handler = stderr_handler
        # Initiate readers for open PIPEs
        if proc.stdout is not None:
            self.stdout_reader = Reader(proc.stdout)
        if proc.stderr is not None:
            self.stderr_reader = Reader(proc.stderr)

    def _stderr_handler(self, line):
        """
        Print line.

        Args:
            line : str
                Line to print.

        """
        print(line.rstrip('\n'))

    def _stdout_handler(self, line):
        """
        Print line.

        Args:
            line : str
                Line to print.

        """
        print(line.rstrip('\n'))

    def process_output(self):
        """
        Loop through all lines that have been written to stdout
        and/or stderr since last call and run through the appropriate
        handlers.

        """
        if self.stdout_reader:
            l = self.stdout_reader.read()
            while l:
                self.stdout_handler(l)
                l = self.stdout_reader.read()
        if self.stderr_reader:
            l = self.stderr_reader.read()
            while l:
                self.stderr_handler(l)
                l = self.stderr_reader.read()