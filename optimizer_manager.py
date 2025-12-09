import opengen as og


class OptimizerManager:
    """Manages multiple TCP optimizer servers."""

    def __init__(self):
        self.managers = []

    def add_optimizer(self, tcp_server_name, port):
        """Add an optimizer TCP manager.

        Args:
            tcp_server_name: Name/path of the optimizer server
            port: TCP port number

        Returns:
            The created optimizer manager
        """
        mng = og.tcp.OptimizerTcpManager(tcp_server_name.replace(".", "_"), port=port)
        self.managers.append(mng)
        return mng

    def start_all(self):
        """Start all registered optimizer servers."""
        for mng in self.managers:
            mng.start()

    def kill_all(self):
        """Stop all registered optimizer servers."""
        for mng in self.managers:
            mng.kill()

    def __enter__(self):
        """Context manager entry - start all servers."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - kill all servers."""
        self.kill_all()
