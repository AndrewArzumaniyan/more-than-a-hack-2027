"""Fixed tracking-only possession inference. No training data or GT required."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from features import features,candidate_design
from model import load_model,predict
from postprocess import apply

HERE=Path(__file__).resolve().parent

def halves(tracking,metadata,cfg):
    xml=ET.parse(metadata).getroot()
    team_ids=[t.get('id') for t in xml.iter('Team')]
    assert len(team_ids)==2
    teams=np.array([0 if p.get('teamId')==team_ids[0] else 1 for p in xml.iter('Player')],dtype=np.int8)
    pitch=np.asarray(cfg['pitch_m'],dtype=np.float32)
    bounds=[(0,cfg['second_half_frame']),(cfg['second_half_frame'],cfg['frames'])]
    ids=[np.arange(a,b,5) for a,b in bounds]
    values=[np.empty((len(f),len(teams)+1,2),np.float32) for f in ids]
    with tracking.open() as stream:
        for frame,line in enumerate(stream):
            h=int(frame>=bounds[1][0]);offset=frame-bounds[h][0]
            if offset%5:continue
            index,players,ball=line.strip().split(':')
            row=np.fromstring(players.replace(';',',')+','+ball,sep=',')
            assert int(index)==frame+1 and len(row)==3*(len(teams)+1)
            values[h][offset//5]=row.reshape(-1,3)[:,:2]*pitch
    assert frame+1==cfg['frames']
    return [SimpleNamespace(frame=f,players=v[:,:-1],teams=teams,ball=v[:,-1],pitch=pitch,
            labels=np.full(len(f),3,dtype=np.int8)) for f,v in zip(ids,values)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('DS Team AI4Football.csv'))
    args=parser.parse_args();cfg=json.loads((HERE/'config.json').read_text())
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    blocks=halves(args.tracking,args.metadata,cfg)
    views=[features(h) for h in blocks]
    contexts=[candidate_design(v) for v in views]
    probabilities=[]
    for member in cfg['models']:
        network=load_model(HERE/member['file'],member['width'])
        probabilities.append([predict(network,h,c,member['points']) for h,c in zip(blocks,contexts)])
        del network
    labels=np.empty(cfg['frames'],np.int8)
    for i,(half,view) in enumerate(zip(blocks,views)):
        p=np.einsum('m,mnc->nc',[m['weight'] for m in cfg['models']],np.stack([p[i] for p in probabilities]))
        states=apply(p,half.ball,view.exit_margin)
        left=0 if i==0 else cfg['second_half_frame'];right=cfg['second_half_frame'] if i==0 else cfg['frames']
        frames=np.arange(left,right);where=np.searchsorted(half.frame,frames,side='right')-1
        labels[frames]=states[where]
    result=pd.DataFrame(dict(frame=np.arange(len(labels)),label=np.array(['H','A','D'])[labels]))
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f'{args.output}: {len(result)} rows; SHA256={digest}',flush=True)
    if digest!=cfg['expected_csv_sha256']:
        raise RuntimeError('Prediction differs from the fixed release; check input files and dependency versions')

if __name__=='__main__':main()
