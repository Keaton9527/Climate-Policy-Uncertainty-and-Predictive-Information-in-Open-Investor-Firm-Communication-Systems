"""Generate absolute paths for a checkout without editing Python source."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--labels');p.add_argument('--qa');p.add_argument('--work-dir');args=p.parse_args()
    root=Path(args.root)
    if not root.is_absolute():raise ValueError('--root must be absolute')
    target=root/'configs/workflow.local.json'
    c=json.loads((root/'configs/workflow.json').read_text(encoding='utf-8'))
    for key,value in {'labels':args.labels or str(root/'private/annotations.xlsx'),'qa':args.qa or str(root/'private/qa.csv'),'work_dir':args.work_dir or str(root/'work/run_001')}.items():
        if not Path(value).is_absolute():raise ValueError(f'{key} must be absolute')
        c[key]=value
    target.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8');print(target)


if __name__=='__main__':main()
