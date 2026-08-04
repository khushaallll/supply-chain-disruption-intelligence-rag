import subprocess

# search the whole project tree for anything with "override" in the name
result = subprocess.run(
    ["find", ".", "-iname", "*override*"],
    capture_output=True, text=True
)
print(result.stdout or "(no matches found)")

# also check if master_enrichment_raw.csv exists, since that one should
# definitely be there (it's referenced as an input in multiple places)
result2 = subprocess.run(
    ["find", ".", "-iname", "*enrichment*"],
    capture_output=True, text=True
)
print(result2.stdout or "(no matches found)")