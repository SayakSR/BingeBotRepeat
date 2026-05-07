Quantitative Dataset — Binge Bot Repeat
========================================

This folder holds the **quantitative release** of the SQLite dataset used for Section
*Large-Scale Measurement of the Telegram Piracy Ecosystem* in the paper *Binge, Bot,
Repeat: Unpacking the Ecosystem of Video Piracy on Telegram*.

Sharding layout
---------------

The database is provided as binary **shards** that must be merged locally before use.

Files:

  quantitative_dataset.sqlite.part001
  quantitative_dataset.sqlite.part002
  …
  quantitative_dataset.sqlite.part008   

Merge instructions (Python)
----------------------------
Requires **Python 3** (standard library only).

From this directory:

  python3 merge_dataset.py

This writes **quantitative_dataset.sqlite** in your **current working directory** (not
automatically beside the shards). Example:

  cd dataset
  python3 merge_dataset.py -o ./quantitative_dataset.sqlite

Options:

  -d / --shards-dir   Directory containing `.partNNN` files  
                      (default: folder where merge_dataset.py lives).

  -o / --output       Output path for the merged database  
                      (default: ./quantitative_dataset.sqlite).

Example when your shell is elsewhere:

  python3 path/to/dataset/merge_dataset.py \
      -d path/to/dataset \
      -o ~/Downloads/quantitative_dataset.sqlite


Sanity check after merge
------------------------
With the sqlite3 CLI installed:

  sqlite3 quantitative_dataset.sqlite "PRAGMA integrity_check;"
  sqlite3 quantitative_dataset.sqlite "SELECT COUNT(*) FROM posts;"

You should see `ok` from integrity_check.


Dataset contents and redaction (merged file)
--------------------------------------------
After merging, you have a single SQLite table **posts**:

  channel_random_id   Opaque random hex id per scraped source channel (no stored public
                        channel URL or handle).
  post_id               Message id from the original Telegram URL (e.g. …/265 → 265).
                        With channel_random_id forms a dataset-local key only.
  post_text             Body text after redaction below.
  post_date             Timestamp string (ISO-8601-style in this build).
  views                 View count (text).
  reactions             Reactions payload (text).


**Removed**

  - Direct channel URLs and full Telegram post URLs.

**Redacted inside post_text**

  1) Corpus-derived Telegram **handles** from channel/post URL paths:
        <Channel name REDACTED>
        <Bot name REDACTED>   — when lowercase handle ends with “bot” (heuristic)

  2) Detected **URLs / links**:
        <URL REDACTED>




