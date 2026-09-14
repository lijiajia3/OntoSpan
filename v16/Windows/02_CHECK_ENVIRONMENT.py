"""Read-only environment/model-cache check. No smoke training."""
if __name__=='__main__':
    from windows_launcher import ensure_runtime
    ensure_runtime()
    from runner import preflight
    preflight(download=False)
