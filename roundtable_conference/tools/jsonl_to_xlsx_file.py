import pandas as pd

# 读取 jsonl 文件
df = pd.read_json("roundtable_conference/output/2001 - beyond/logs/run_summary.jsonl", lines=True)

# 直接导出为 xlsx
df.to_excel("output/2001 - beyond/output.xlsx", index=False)