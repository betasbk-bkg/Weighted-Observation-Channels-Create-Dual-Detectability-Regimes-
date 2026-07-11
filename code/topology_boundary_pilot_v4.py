"""
control-topology detectability pilot v4 (self-contained, numpy만 필요)
실행:  python topology_boundary_pilot_v4.py [MC] [REPS]
       (default MC=200, REPS=1;  예: python topology_boundary_pilot_v4.py 100 3)

v3 대비 수정 (2026-07-05):
  1. p-grid 상한 0.45 → 0.50 (과반 경계 마감 — "준과반 전 구간" claim 완결용)
  2. c-sweep 3점 → 6점 [0, 0.2, 0.4, 0.6, 0.8, 1.0]
     (c-기울기 반전의 형상 확정 + opposite 가시→비가시 전환 임계 c† 탐색)
  3. MC 기본 200 (CLI 인자로 조정 가능 — 무거우면 100부터)
  4. REPS: seed 복제 (p* 산포 확인용, REPS>=3이면 mean±sd 출력)
  5. 결과 CSV 저장 (topology_v4_results.csv) — 논문 그림/피팅 재사용
  6. 진행상황 출력 (combo별 경과시간)

voter 모델·가중 C 수식·탐지기(CUSUM)·보간·NaN 방향판정은 v3와 동일.
"""
import numpy as np, sys, time, csv
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
    if topology=='D':
        k=max(1,round(0.1*len(votes))); sub=rng.choice(len(votes),k,replace=False)
        return g, DIRS[votes[sub]].mean(axis=0)
    if topology=='C':
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

def obs_pstar(mode,c,topology,tgt,pg,MC,rng,early_stop=True):
    tprs=[]; rmses=[]
    for j,p in enumerate(pg):
        h=0; rr=[]
        for _ in range(MC):
            g,rm=run_trace(p,mode,c,topology,rng); h+=detect(g); rr.append(rm)
        tprs.append(h/MC); rmses.append(np.mean(rr))
        if early_stop and tprs[-1]>=tgt: break
    tprs=np.array(tprs); scanned=pg[:len(tprs)]
    if tprs[-1]<tgt:  return np.nan, np.mean(rmses), tprs, 'above'
    if tprs[0]>=tgt:  return scanned[0], np.mean(rmses), tprs, 'below'
    i=len(tprs)-1
    ps=scanned[i-1]+(tgt-tprs[i-1])*(scanned[i]-scanned[i-1])/(tprs[i]-tprs[i-1])
    return ps, np.mean(rmses), tprs, 'in'

def fmt(ps,status,pg):
    if status=='above': return f">{pg[-1]:.2f}"
    if status=='below': return f"<={ps:.3f}"
    return f"{ps:.4f}"

if __name__=="__main__":
    MC=int(sys.argv[1]) if len(sys.argv)>1 else 200
    REPS=int(sys.argv[2]) if len(sys.argv)>2 else 1
    PG=[0.055,0.06,0.065,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
    CS=[0.0,0.2,0.4,0.6,0.8,1.0]; TGT=0.8
    t_start=time.time()
    print("="*66)
    print(f"topology-boundary pilot v4 | N={N_AG} MC={MC} REPS={REPS} tgt={TGT}")
    print(f"p_grid(상한 0.50)={PG}")
    print(f"c_sweep={CS}")
    print("="*66)
    rows=[]

    print("\n### T1: F vs D (v3와 동일 — 연속성 확인용) ###")
    print(f"{'c':>4} {'p*_F':>8} {'p*_D':>8} {'|d|':>7}")
    t1=True
    for c in [0.0,0.4,0.8]:
        r=np.random.default_rng(10+int(c*10)); pf,_,_,sf=obs_pstar('opposite',c,'F',TGT,PG,MC,r)
        r=np.random.default_rng(10+int(c*10)); pd_,_,_,sd_=obs_pstar('opposite',c,'D',TGT,PG,MC,r)
        d=abs(pf-pd_) if pf==pf and pd_==pd_ else float('nan')
        print(f"{c:>4} {fmt(pf,sf,PG):>8} {fmt(pd_,sd_,PG):>8} {d:>7.4f}")
        if d==d and d>0.015: t1=False
    print(f"T1: {'PASS' if t1 else 'FAIL'}")

    print("\n### T3: c-sweep — F vs C, opposite/scatter (핵심) ###")
    print(f"{'mode':>9} {'c':>4} {'rep':>4} {'p*_F':>8} {'p*_C':>8}  TPR_C@top4")
    for mode in ['opposite','scatter']:
        for c in CS:
            pfs=[]; pcs=[]; scs=[]
            for rep in range(REPS):
                seed=100+rep*1000+int(c*10)+(0 if mode=='opposite' else 500)
                r=np.random.default_rng(seed); pf,_,_,sf=obs_pstar(mode,c,'F',TGT,PG,MC,r)
                r=np.random.default_rng(seed); pc,_,tc,sc=obs_pstar(mode,c,'C',TGT,PG,MC,r)
                pfs.append(pf); pcs.append(pc); scs.append(sc)
                tstr=" ".join(f"{x:.2f}" for x in tc[-4:])
                el=time.time()-t_start
                print(f"{mode:>9} {c:>4} {rep:>4} {fmt(pf,sf,PG):>8} {fmt(pc,sc,PG):>8}  [{tstr}]  ({el:.0f}s)")
                rows.append([mode,c,'F',rep,pf,sf]); rows.append([mode,c,'C',rep,pc,sc])
            if REPS>=3:
                a=np.array(pcs,float)
                if np.isfinite(a).all():
                    print(f"{'':>9} {c:>4} {'m±s':>4} {'':>8} {np.mean(a):>8.4f}  (sd={np.std(a):.4f})")

    with open('topology_v4_results.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['mode','c','topology','rep','pstar','status'])
        for row in rows: w.writerow(row)
    print(f"\nCSV 저장: topology_v4_results.csv | 총 경과 {time.time()-t_start:.0f}s")
    print("판정 지침: opposite에서 c† = p*_C가 'in'→'above'로 넘어가는 c 구간.")
    print("           scatter에서 p*_C(c) 하강 형상(선형/포화) — 유도 노트와 대조.")
    print("="*66)
