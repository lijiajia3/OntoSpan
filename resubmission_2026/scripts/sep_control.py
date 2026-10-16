"""Separation control (see PREREG_separation_control.md): implicit term, embeddings at 1e-3, lambda 0/1/10, ten seeds."""
import time
import diag_lambda_law as D
import worker

def main():
    D.OUT.mkdir(parents=True, exist_ok=True)
    for lam in (0.0, 1.0, 10.0):
        for sd in (42, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            j = D.job(sd, lam, 1e-3)
            out = D.OUT / j["id"]
            if (out / "COMPLETE.json").exists():
                continue
            D.CUR["lr_type"] = 1e-3
            t = time.time()
            r = worker.train(j, out, device="cuda")
            m = r["metrics"]
            print("DONE", j["id"], "F1=%.5f" % m["F1"], "%.0fs" % (time.time() - t), flush=True)
            (out / "best.pt").unlink(missing_ok=True)

if __name__ == "__main__":
    main()
