"""
control-topology detectability pilot v3 (self-contained, numpy만 필요)
실행:  python topology_boundary_pilot_v3.py

v2 대비 수정 (2026-07-05, 검증 기반):
  1. p-grid 상한 0.15 → 0.45 확장 (v2의 NaN = p*_C가 grid 위 — 관측창 개방)
     TPR 타깃은 0.8 유지: p* 정의 보존 → F/D/C가 동일 좌표계에서 비교 가능.
  2. NaN 방향판정 추가: v2는 NaN을 "효과 미미"로 오판정.
     v3는 TPR 궤적으로 구분 —
       max(TPR) < tgt        → p* > grid 상한 (강한 억제, censored above)
       TPR[0]  >= tgt        → p* <= grid 하한 (censored below)
  3. 조기중단: 첫 교차 확보 시 스캔 중지 (보간에 필요한 건 교차 전후 2점뿐,
     TPR이 p에 단조증가라는 가정 하에 p* 정의 불변·시간 절약)
  4. 판정문 교체: censored 상태를 명시 출력, '미미' 자동판정 제거.

voter 모델·가중 C 수식·탐지기(CUSUM)·보간은 v2와 동일 (변경 없음).
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

def obs_pstar(mode,c,topology,tgt,pg,MC,rng,early_stop=True):
    """returns (p*, mean_rmse_scanned, tprs_scanned, status)
    status: 'in' (보간됨) | 'below' (p*<=pg[0]) | 'above' (p*>마지막 스캔점) """
    tprs=[]; rmses=[]
    for j,p in enumerate(pg):
        h=0; rr=[]
        for _ in range(MC):
            g,rm=run_trace(p,mode,c,topology,rng); h+=detect(g); rr.append(rm)
        tprs.append(h/MC); rmses.append(np.mean(rr))
        if early_stop and tprs[-1]>=tgt: break
    tprs=np.array(tprs); scanned=pg[:len(tprs)]
    if tprs[-1]<tgt:                      # 끝까지 미달 → grid 위 (censored above)
        return np.nan, np.mean(rmses), tprs, 'above'
    if tprs[0]>=tgt:                      # 첫 점부터 초과 → grid 아래 (censored below)
        return scanned[0], np.mean(rmses), tprs, 'below'
    i=len(tprs)-1                         # 조기중단 시 마지막 점이 첫 교차
    ps=scanned[i-1]+(tgt-tprs[i-1])*(scanned[i]-scanned[i-1])/(tprs[i]-tprs[i-1])
    return ps, np.mean(rmses), tprs, 'in'

def fmt(ps,status,pg):
    if status=='above': return f">{pg[-1]:.2f}"
    if status=='below': return f"<={ps:.3f}"
    return f"{ps:.4f}"

if __name__=="__main__":
    PG=[0.055,0.06,0.065,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
    MC=80; TGT=0.8
    print("="*64)
    print(f"topology-boundary pilot v3 | N={N_AG} MC={MC} tgt={TGT}")
    print(f"p_grid={PG}")
    print("="*64)

    print("\n### T1: F vs D (p* 등식? RMSE 단순 '다름'만 확인) ###")
    print(f"{'c':>4} {'p*_F':>8} {'p*_D':>8} {'|d|':>7} {'RMSE_F':>7} {'RMSE_D':>7}")
    t1=True
    for c in [0.0,0.4,0.8]:
        r=np.random.default_rng(10+int(c*10)); pf,rf,_,sf=obs_pstar('opposite',c,'F',TGT,PG,MC,r)
        r=np.random.default_rng(10+int(c*10)); pd_,rd,_,sd_=obs_pstar('opposite',c,'D',TGT,PG,MC,r)
        d=abs(pf-pd_) if pf==pf and pd_==pd_ else float('nan')
        print(f"{c:>4} {fmt(pf,sf,PG):>8} {fmt(pd_,sd_,PG):>8} {d:>7.4f} {rf:>7.4f} {rd:>7.4f}")
        if d==d and d>0.015: t1=False
    print(f"T1: {'PASS (p* 등식, RMSE는 다를 수 있음)' if t1 else 'FAIL'}")

    print("\n### T3: 가중 C의 p*_C 실측 (확장 grid, 동일 TPR 기준) ###")
    print(f"{'mode':>9} {'c':>4} {'p*_F':>8} {'p*_C':>8} {'C-F':>8}  TPR_C@grid")
    rows={'opposite':[], 'scatter':[]}
    for mode in ['opposite','scatter']:
        for c in [0.0,0.4,0.8]:
            r=np.random.default_rng(20+int(c*10)); pf,_,_,sf=obs_pstar(mode,c,'F',TGT,PG,MC,r)
            r=np.random.default_rng(20+int(c*10)); pc,_,tc,sc=obs_pstar(mode,c,'C',TGT,PG,MC,r)
            df=pc-pf if pf==pf and pc==pc else float('nan')
            rows[mode].append((c,pf,sf,pc,sc,df))
            tstr=" ".join(f"{x:.2f}" for x in tc[-4:])
            print(f"{mode:>9} {c:>4} {fmt(pf,sf,PG):>8} {fmt(pc,sc,PG):>8} {df:>+8.4f}  [...{tstr}]")
    # 판정 — censored 명시, '미미' 자동판정 제거
    print("\n  판정:")
    any_above=any(s=='above' for m in rows for _,_,_,_,s,_ in rows[m])
    all_above=all(s=='above' for m in rows for _,_,_,_,s,_ in rows[m])
    if all_above:
        print(f"  → 전 조건 censored above: p*_C > {PG[-1]} — 가중 C는 opposite·scatter 모두를")
        print(f"    grid 상한 밖으로 억제. '효과 미미' 아님(정반대). H1(단조 강억제) 방향이나,")
        print(f"    상한을 더 열어야 p*_C 존재/부재가 확정됨.")
    elif any_above:
        print("  → 일부 censored: mode/c에 따라 p*_C 진입 여부가 갈림 — 차등 구조. 아래 표로 형상 판단.")
    else:
        print("  → 전 조건에서 p*_C 실측됨: H1/H2는 c-궤적 형상(단조 vs 급변)으로 판정.")

    print("\n" + "="*64)
    print("다음: p*_C 형상 확정 → M/S 위상 포함 full grid(MC=200)")
    print("="*64)
