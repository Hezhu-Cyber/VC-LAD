"""Generate a clearly synthetic background and exercise the runnable pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
from run import execute
from vclad.detector import CONTINUOUS_COLUMNS
if __name__=='__main__':
    rng=np.random.default_rng(42)
    n=800; root=Path('outputs/demo');root.mkdir(parents=True,exist_ok=True)
    values=np.array([35,350,12,65,10000,3.7,3.65,30,25])+rng.normal(0,.01,(n,9))
    frame=pd.DataFrame(values,columns=CONTINUOUS_COLUMNS)
    frame['数据时间']=pd.date_range('2000-01-01',periods=n,freq='10s')
    frame['最高报警等级']=0;frame['车辆状态']=1;frame['充电状态']=3;frame['挡位']=13
    frame.to_csv(root/'synthetic.csv',index=False)
    execute(root/'synthetic.csv',root/'run',epochs=1)
