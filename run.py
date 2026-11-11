"""Runnable per-vehicle training/inference example for the public subset.
Raw CSV files are unchanged. No fault ground truth is inferred from alarm levels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from vclad import detector as d
from vclad.preprocessing import split_healthy
from vclad.ranking import variable_scores

def execute(source, output, epochs=30, batch_size=64, seed=42):
    d.set_seed(seed)
    torch.set_num_threads(2)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    train,test=split_healthy(pd.read_csv(source))
    destination=output/'prepared'/'subset'/'train'
    destination.mkdir(parents=True,exist_ok=True)
    train.to_csv(destination/'train_vehicle.csv',index=False,encoding='utf-8-sig')
    test.to_csv(output/'healthy_test.csv',index=False,encoding='utf-8-sig')
    settings=d.TrainSettings(epochs=epochs,batch_size=batch_size)
    model_dir=output/'models'
    d.train_vehicle(output/'prepared','subset','vehicle',model_dir,settings,torch.device('cpu'),force=True)
    ck=torch.load(d.checkpoint_path(model_dir,'vehicle'),map_location='cpu',weights_only=False)
    model=d.VCLAD();model.load_state_dict(ck['model_state_dict']);model.eval()
    scaler=d.MinMaxScaler.from_state_dict(ck['scaler'])
    results=[]
    with torch.no_grad():
        for sid,group in test.groupby('segment_id',sort=False):
            if len(group)<30: continue
            values=scaler.transform(d._clean_continuous(group,d.CONTINUOUS_COLUMNS))
            windows=d._make_windows(values,30,1)
            for start in range(0,len(windows),128):
                batch=torch.from_numpy(windows[start:start+128])
                scores,_,_=d._score_batch(model,batch,ck['score_stats'],20.0,'last',0.8,0.2)
                contrib=variable_scores(batch,model(batch)['x_recon'])
                for offset,score in enumerate(scores.numpy()):
                    last=start+offset+29
                    results.append({'segment_id':sid,'数据时间':group.iloc[last]['数据时间'],'anomaly_score':float(score),'top_variable':d.CONTINUOUS_COLUMNS[int(contrib[offset].argmax())]})
    if not results: raise ValueError('No test segment supports a 30-record window')
    pd.DataFrame(results).to_csv(output/'scores.csv',index=False,encoding='utf-8-sig')
    report={'train_rows':len(train),'healthy_test_rows':len(test),'test_windows':len(results),'parameters':sum(p.numel() for p in model.parameters()),'epochs':epochs,'seed':seed,'note':'Scores only. No fault labels or performance claims; no post-score smoothing.'}
    (output/'run_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/run'));p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();execute(a.csv,a.output,a.epochs,a.batch_size)
