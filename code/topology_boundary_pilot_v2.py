"""
control-topology detectability pilot v2 (self-contained, numpy만 필요)
실행:  python topology_boundary_pilot_v2.py

v1 대비 수정 (검증 기반):
  1. erf 벡터화 (v1은 문서 지적대로 스칼라 erf였음 — 여기선 np.vectorize)
  2. p-grid 하한 0.055까지 확장 (high-c 검열 제거)
  3. T3를 'c-교차'가 아니라 'attack-type별 가중효과 부호' 측정으로 교체
     + 가중 방향을 열어두고 측정만 (opposite/scatter 각각 p*_C vs p*_F)
  4. T1 판정에서 'RMSE_D > RMSE_F' 강제 제거 → 'RMSE 다름'만 확인 (문서 지적)

이 스크립트는 이론 예측이 아니라 실제 시뮬레이터 관측으로 T1/T3를 검정한다.
"""
import numpy as np
from math import erf as _erf

P0,H,L_H,DELTA,GRID=0.05,14.0,20,0.5,220
WARMUP,LAM,N_AG=15,0.15,120
DIRS=np.array([[np.cos(2*np.pi*k/8),np.sin(2*np.pi*k/8)] for k in range(8)])
_verf=np.vectorize(_erf)
def ncdf(x): return 0.5*(1+_verf(np.asarray(x,float)/np.sqrt(2)))

def angle_to_dir(a): return int(np.round(a/(2*np.pi/8)))%8

def gen_votes(ideal,prev,p,mode,c,rng):
    N=N_AG; n_tr=min(round(N*p),N); n_h=N-n_tr; out=[]
    n_acc=round(n_h*0.7368); n_slow=round(n_h*0.2105); n_oth=n_h-n_acc-n_slow
    for _ in range(n_acc): out.append(angle_to_dir(ideal+np.deg2rad(rng.uniform(-3,3))))
    for _ in range(n_slow):
        lag=rng.uniform(0.2,0.5); a=prev+(ideal-prev)*(1-lag); out.append(angle_to_dir(a))
    for _ in range(max(n_oth,0)): out.append(angle_to_dir(ideal+np.deg2rad(rng.uniform(-30,30))))
    n_al=int(round(c*n_tr)); base=angle_to_dir(ideal+np.pi)
    if mode=='opposite':
        for _ in range(n_al): out.append((base+rng.integers(-1,2))%8)
    else:  # scatter
        sup=rng.choice(8,3,replace=False)
        for _ in range(n_al): out.append(int(sup[rng.integers(0,3)]))
    for _ in range(n_tr-n_al): out.append(int(rng.integers(0,8)))
    return np.array(out[:len(out)])

def gamma_and_cmd(votes, topology, rng):
    V=DIRS[votes]; m=V.mean(axis=0); g=np.linalg.norm(m)
    if topology=='F':
        return g, m
    if topology=='D':          # 위임: k명 추첨 실행, gamma는 전원(O_T 불변)
        k=max(1,round(0.1*len(votes))); sub=rng.choice(len(votes),k,replace=False)
        return g, DIRS[votes[sub]].mean(axis=0)
    if topology=='C':          # 가중협력: gamma도 가중으로 관측(O_T 변함)
        mhat=m/(np.linalg.norm(m)+1e-9); w=np.clip(V@mhat,0,None)**2
        cw=(V*w[:,None]).sum(axis=0)/(w.sum()+1e-9)
        return np.linalg.norm(cw), cw

def run_trace(p,mode,c,topology,rng,T=60,onset=20):
    ideal=0.0; prev=0.0; gs=[]; ee=[]; tgt=np.array([1.0,0.0])
    for t in range(T):
        atk=(t>=onset); pp=p if atk else P0; cc=c if atk else 0.0; mm=mode if atk else 'opposite'
        vv=gen_votes(ideal,prev,pp,mm,cc,rng)
        g,cmd=gamma_and_cmd(vv,topology,rng)
        gs.append(g); ee.append(np.linalg.norm(cmd-tgt)); prev=ideal
    return np.array(gs), np.mean(ee[onset:])

def detect(gs,onset=20):
    mu=gs[:WARMUP].mean(); var=max(gs[:WARMUP].var(),1e-4); cs=0.0
    for i in range(WARMUP,len(gs)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gs[i])/sd; cs=max(0.0,cs+dev-DELTA)
        if i>=onset and cs>H: return 1 if (i-onset)<L_H else 0
        if cs<H*0.5: mu=(1-LAM)*mu+LAM*gs[i]; var=(1-LAM)*var+LAM*(gs[i]-mu)**2
    return 0

def obs_pstar(mode,c,topology,tgt,pg,MC,rng):
    tprs=[]; rmses=[]
    for p in pg:
        h=0; rr=[]
        for _ in range(MC):
            g,rm=run_trace(p,mode,c,topology,rng); h+=detect(g); rr.append(rm)
        tprs.append(h/MC); rmses.append(np.mean(rr))
    tprs=np.array(tprs); idx=np.where(tprs>=tgt)[0]
    if len(idx)==0: ps=np.nan
    elif idx[0]==0: ps=pg[0]
    else:
        i=idx[0]; ps=pg[i-1]+(tgt-tprs[i-1])*(pg[i]-pg[i-1])/(tprs[i]-tprs[i-1])
    return ps, np.mean(rmses)

if __name__=="__main__":
    PG=[0.055,0.06,0.065,0.07,0.08,0.09,0.10,0.12,0.15]   # 하한 확장
    MC=80; TGT=0.8
    print("="*64)
    print(f"topology-boundary pilot v2 | N={N_AG} MC={MC} tgt={TGT}")
    print(f"p_grid={PG}")
    print("="*64)

    print("\n### T1: F vs D (p* 등식? RMSE 단순 '다름'만 확인) ###")
    print(f"{'c':>4} {'p*_F':>7} {'p*_D':>7} {'|d|':>7} {'RMSE_F':>7} {'RMSE_D':>7}")
    t1=True
    for c in [0.0,0.4,0.8]:
        r=np.random.default_rng(10+int(c*10)); pf,rf=obs_pstar('opposite',c,'F',TGT,PG,MC,r)
        r=np.random.default_rng(10+int(c*10)); pd_,rd=obs_pstar('opposite',c,'D',TGT,PG,MC,r)
        d=abs(pf-pd_) if pf==pf and pd_==pd_ else float('nan')
        print(f"{c:>4} {pf:>7.4f} {pd_:>7.4f} {d:>7.4f} {rf:>7.4f} {rd:>7.4f}")
        if d==d and d>0.015: t1=False
    print(f"T1: {'PASS (p* 등식, RMSE는 다를 수 있음)' if t1 else 'FAIL'}")

    print("\n### T3(재정의): 가중 C의 효과를 attack-mode별로 측정 ###")
    print("  측정: p*_C - p*_F 부호. opposite/scatter 각각.")
    print(f"{'mode':>9} {'c':>4} {'p*_F':>7} {'p*_C':>7} {'C-F':>8}")
    signs={'opposite':[], 'scatter':[]}
    for mode in ['opposite','scatter']:
        for c in [0.0,0.4,0.8]:
            r=np.random.default_rng(20+int(c*10)); pf,_=obs_pstar(mode,c,'F',TGT,PG,MC,r)
            r=np.random.default_rng(20+int(c*10)); pc,_=obs_pstar(mode,c,'C',TGT,PG,MC,r)
            df=pc-pf if pf==pf and pc==pc else float('nan')
            signs[mode].append(df)
            print(f"{mode:>9} {c:>4} {pf:>7.4f} {pc:>7.4f} {df:>+8.4f}")
    # 판정: attack-type별로 부호가 갈리거나(원래 T3'), 정도차가 나거나
    opp_mean=np.nanmean(signs['opposite']); sc_mean=np.nanmean(signs['scatter'])
    print(f"\n  평균 C-F: opposite={opp_mean:+.4f}, scatter={sc_mean:+.4f}")
    if opp_mean*sc_mean<0:
        print("  → T3'(부호분리): attack-type에 따라 가중효과 부호가 갈림. 살림.")
    elif abs(opp_mean-sc_mean)>0.01:
        print("  → T3''(정도차): 양쪽 같은 부호지만 크기 차등. 재정식화 후보.")
    else:
        print("  → 가중효과 미미: C 위상을 headline에서 제외 검토.")

    print("\n" + "="*64)
    print("다음: T1 PASS + T3 부호/정도 패턴 확인되면 M/S 포함 full grid(MC=200)")
    print("="*64)
