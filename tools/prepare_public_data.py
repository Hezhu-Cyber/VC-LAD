"""Project only approved fields; never overwrite input workbooks."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, date
from pathlib import Path
from openpyxl import load_workbook

FIELDS = ['数据时间','最高报警等级','车速','总电压','总电流','SOC','绝缘电阻','电池单体电压最高值','电池单体电压最低值','最高温度值','最低温度值','车辆状态','充电状态','挡位']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input-dir',type=Path,required=True)
    a.add_argument('--output-dir',type=Path,required=True)
    args=a.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.input_dir.glob('*.xlsx'),key=lambda p:int(p.stem))
    if len(paths)!=10: raise ValueError('Expected exactly ten numbered XLSX files')
    report=[]
    for n,p in enumerate(paths,1):
        before=digest(p)
        w=load_workbook(p,read_only=True,data_only=True)
        s=w['车辆数据']; rows=s.iter_rows(values_only=True); headers=list(next(rows))
        indices=[headers.index(k) for k in FIELDS]
        target=args.output_dir/f'vehicle_{n:03d}.csv'
        if target.exists(): raise FileExistsError(target)
        count=0; missing={k:0 for k in FIELDS}
        with target.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(FIELDS)
            for row in rows:
                selected=[row[i] for i in indices]
                for k,v in zip(FIELDS,selected): missing[k]+=int(v is None or v=='')
                writer.writerow([v.isoformat(sep=' ') if isinstance(v,datetime) else v.isoformat() if isinstance(v,date) else v for v in selected])
                count+=1
        w.close()
        assert digest(p)==before,'Input was changed'
        report.append(dict(file=target.name,rows=count,columns=len(FIELDS),source_columns=len(headers),removed_fields=[k for k in headers if k not in FIELDS],missing_values=missing,sha256=digest(target),source_unchanged=True))
        print(target.name,count,'rows',flush=True)
    (args.output_dir.parent/'manifest.json').write_text(json.dumps(dict(fields=FIELDS,vehicles=report,total_rows=sum(r['rows'] for r in report),policy='Column projection only. No row filtering, imputation, normalization, label creation, or timestamp shifting. Only the vehicle-data sheet is exported.'),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
