"""Full main table. Completed overlapping jobs are validated then skipped."""
if __name__=='__main__':
    from windows_launcher import ensure_runtime
    ensure_runtime()
    from runner import main
    main('main')
