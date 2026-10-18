# Temp-File and Canary Disk-Search Check
<!-- Member 2 evidence – fill after running scripts/canary_disk_search.sh -->

> **Instructions:** Run `bash scripts/canary_disk_search.sh` after completing the offline demo.
> Paste the script output below.  Never write PASSED for a check you did not actually run.

| Field             | Value |
|-------------------|-------|
| Date/time         | _(fill in)_ |
| Commit hash       | _(fill in: `git rev-parse --short HEAD`)_ |
| Canary phrase     | _(fill in: unique phrase embedded in test document)_ |
| Demo document     | _(fill in: filename used for the canary run)_ |
| Machine / OS      | _(fill in)_ |

## Directories searched

- [ ] `/tmp`
- [ ] `$HOME` (recursively, excluding binary files)
- [ ] `~/.streamlit/`
- [ ] `~/.ollama/` (model weights dir – only text files)
- [ ] Project tree (`.`, excluding `.git/` and the canary file itself)

## Grep results

```
(paste raw grep output here)
```

**Unexpected matches found:** _(Yes / No)_

If Yes, list each match and explain why it is or is not a privacy concern:

| File | Reason for match | Action taken |
|------|-----------------|--------------|
| …    | …               | …            |

## Browser network tab check

During page load, open DevTools → Network tab. Confirm all requests go to `localhost`
or `127.0.0.1` only.

**External requests observed:** _(Yes / No)_

Screenshot or notes:
_(attach or describe)_

## Result

- [ ] PASS – canary phrase not found outside expected locations; no external browser requests
- [ ] FAIL – see matches above; investigation required

_Completed by:_ _(name)_  
_Date:_ _(date)_
