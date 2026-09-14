"""All 120 supplementary reviewer configurations, excluding the 54 main jobs."""
if __name__=='__main__':
    from windows_launcher import ensure_runtime
    ensure_runtime()
    from runner import main
    main('review')
