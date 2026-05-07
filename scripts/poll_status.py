"""Poll the leaderboard for our group's submission status."""
from __future__ import annotations

import argparse
import json
import sys
import time
from gradio_client import Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="36")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--space", default="https://cis4190-newsheadlineclassifier.hf.space")
    args = ap.parse_args()

    c = Client(args.space, verbose=False)
    while True:
        try:
            result = c.predict(args.group, api_name="/check_submission_status")
        except Exception as e:
            print(f"poll error: {type(e).__name__}: {e}")
            if args.once:
                sys.exit(1)
            time.sleep(args.every)
            continue
        # result is (md_success_header, df_success, md_fail_header, df_fail)
        md_succ, df_succ, md_fail, df_fail = result
        ts = time.strftime("%H:%M:%S")
        succ_rows = df_succ.get("data", []) if isinstance(df_succ, dict) else []
        fail_rows = df_fail.get("data", []) if isinstance(df_fail, dict) else []
        print(f"[{ts}] Group {args.group}: {len(succ_rows)} successful, {len(fail_rows)} failed")
        # Show recent rows
        if succ_rows:
            print("  Successful (most recent first):")
            for row in succ_rows[:30]:
                print(f"    {row}")
        if fail_rows:
            print("  Failed (most recent first):")
            for row in fail_rows[:3]:
                # Truncate long error messages
                row_short = [str(c)[:200] for c in row]
                print(f"    {row_short}")
        if args.once:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
