"""
control-topology detectability pilot v5 (self-contained, numpy만)
실행:  python topology_boundary_pilot_v5.py [MC] [REPS]
       예) python topology_boundary_pilot_v5.py 200 1   (기본 MC=200, REPS=1)
           python topology_boundary_pilot_v5.py 120 3   (복제 3회, p* 산포)

계보: v4(F/D/C, γ-CUSUM 검증됨) + 업로드 zip(다위상 골격)의 통합·수정판.
zip 파일럿 v0의 4문제 수정:
  [문제1] p* proxy 붕괴(전 위상 TPR=1.0@p=0.1, 경계 미해상):
          → v4의 시계열 γ-CUSUM 탐지기 채택(과민한 5-특징 합산 score 폐기).
             경계가 격자 안에 들어오도록 grid를 저분율(0.055~)부터.
  [문제2] C_obs가 '탐지가시성 가중'(공격 노출↑)으로 내 발견과 반대:
          → C를 clip-코사인 '명령·γ 재가중'(w=max(V·m̂,0)^2)으로 정의(정리 A 대상).
             zip의 C_obs(가시성 가중)는 C_vis로 별도 보존(대조군).
  [문제3] S(hold)의 RMSE 168 폭발:
          → hold 후 정규화·발산가드 추가. S는 F별칭(분산=현행)이 아니라
             'supervised hold'로만 정의(zip 정의 따름), 가드로 폭발 차단.
  [문제4] MC=6: 기본 200, grid 조밀, 조기중단으로 비용 억제.

★ 나-2(M_axis): M을 '성분별 관측 γ = min(|Vx.mean|,|Vy.mean|)'로 정의 —
   약한 축을 노출하는 새 탐지축. M_perf(성분평균 실행, γ=|m|=F동일)와 분리해
   두 컬럼으로 동시 측정 → '축별 관측이 새 탐지채널인가'를 데이터로 판정.

위상 집합: F, D, C(clip재가중), C_vis(zip식 가시성가중=대조), M_perf, M_axis, S(hold+가드)
attack mode: opposite, scatter
"""
import numpy as np, sys, time, csv
from math import erf as _erf

P0,H,L_H,DELTA=0.05,14.0,20,0.5
WARMUP,LAM,N_AG=15,0.15,120
DIRS=np.array([[np.cos(2*np.pi*k/8),np.sin(2*np.pi*k/8)] for k in range(8)])
_verf=np.vectorize(_erf)
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
    else:
        sup=rng.choice(8,3,replace=False)
        for _ in range(n_al): out.append(int(sup[rng.integers(0,3)]))
    for _ in range(n_tr-n_al): out.append(int(rng.integers(0,8)))
    return np.array(out[:len(out)])

def delegate_mask(n): 
    k=max(1,round(0.1*n)); return np.arange(n)<k  # 결정적(index 기반, seed정합)

def gamma_and_cmd(votes, topology, prev_cmd, rng):
    V=DIRS[votes]; m=V.mean(axis=0); g=np.linalg.norm(m)
    if topology=='F':
        return g, m
    if topology=='D':
        k=max(1,round(0.1*len(votes))); sub=rng.choice(len(votes),k,replace=False)
        return g, DIRS[votes[sub]].mean(axis=0)
    if topology=='C':                      # [수정2] clip-코사인 명령·γ 재가중 (정리 A 대상)
        mhat=m/(np.linalg.norm(m)+1e-9); w=np.clip(V@mhat,0,None)**2
        cw=(V*w[:,None]).sum(axis=0)/(w.sum()+1e-9)
        return np.linalg.norm(cw), cw
    if topology=='C_vis':                  # zip식: anti-intended 가시성 가중 (탐지 관측만, 실행=F)
        anti=np.clip(-(V@np.array([1.0,0.0])),0,None)  # 반-의도 성분
        wv=1.0+2.0*anti                    # anti에 가중(탐지 γ만 낮춤)
        gv=np.linalg.norm((V*wv[:,None]).sum(0)/wv.sum())
        return gv, m                       # 실행은 평균(F), 관측 γ만 가중
    if topology=='M_perf':                 # 성분평균 실행, 관측 γ=|m| (F와 동일 관측)
        return g, np.array([V[:,0].mean(), V[:,1].mean()])
    if topology=='M_axis':                 # [나-2] 성분별 관측: 약한 축 노출
        gx,gy=abs(V[:,0].mean()),abs(V[:,1].mean())
        g_axis=min(gx,gy)*np.sqrt(2)       # 약축 강조(√2로 스케일 맞춤)
        return g_axis, np.array([V[:,0].mean(), V[:,1].mean()])
    if topology=='S':                      # [수정3] supervised hold + 발산가드
        if g<0.55: 
            cmd=prev_cmd.copy()            # hold
        else:
            cmd=m
        cn=np.linalg.norm(cmd)
        if cn>1e-9: cmd=cmd/cn             # 정규화(폭발 차단)
        return g, cmd
    raise ValueError(topology)

def run_trace(p,mode,c,topology,rng,T=60,onset=20):
    ideal=0.0; prev=0.0; gs=[]; ee=[]; tgt=np.array([1.0,0.0]); prev_cmd=np.array([1.0,0.0])
    for t in range(T):
        atk=(t>=onset); pp=p if atk else P0; cc=c if atk else 0.0; mm=mode if atk else 'opposite'
        vv=gen_votes(ideal,prev,pp,mm,cc,rng)
        g,cmd=gamma_and_cmd(vv,topology,prev_cmd,rng)
        cn=np.linalg.norm(cmd)
        cmd_unit = cmd/cn if cn>1e-9 else prev_cmd
        gs.append(g); ee.append(np.linalg.norm(cmd_unit-tgt)); prev=ideal; prev_cmd=cmd_unit
    return np.array(gs), np.mean(ee[onset:])

def detect(gs,onset=20):                    # [수정1] v4의 시계열 γ-CUSUM (과민 score 폐기)
    mu=gs[:WARMUP].mean(); var=max(gs[:WARMUP].var(),1e-4); cs=0.0
    for i in range(WARMUP,len(gs)):
        sd=np.sqrt(var)+1e-6; dev=(mu-gs[i])/sd; cs=max(0.0,cs+dev-DELTA)
        if i>=onset and cs>H: return 1 if (i-onset)<L_H else 0
        if cs<H*0.5: mu=(1-LAM)*mu+LAM*gs[i]; var=(1-LAM)*var+LAM*(gs[i]-mu)**2
    return 0

def obs_pstar(mode,c,topology,tgt,pg,MC,rng,early_stop=True):
    tprs=[]; rmses=[]
    for p in pg:
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

def fmt(ps,st,pg):
    if st=='above': return f">{pg[-1]:.2f}"
    if st=='below': return f"<={ps:.3f}"
    return f"{ps:.4f}"

if __name__=="__main__":
    MC=int(sys.argv[1]) if len(sys.argv)>1 else 200
    REPS=int(sys.argv[2]) if len(sys.argv)>2 else 1
    PG=[0.055,0.06,0.065,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
    CS=[0.0,0.2,0.4,0.6,0.8,1.0]; TGT=0.8
    TOPOS=['F','D','C','C_vis','M_perf','M_axis','S']
    t0=time.time()
    print("="*70)
    print(f"topology-boundary pilot v5 | N={N_AG} MC={MC} REPS={REPS} tgt={TGT}")
    print(f"topologies={TOPOS}")
    print(f"p_grid(0.055~0.50)  c_sweep={CS}")
    print("="*70)
    rows=[]

    # T1: F vs D (연속성)
    print("\n### T1: F vs D (탐지경계 등식 · RMSE 상이) ###")
    print(f"{'c':>4} {'p*_F':>8} {'p*_D':>8} {'|d|':>7} {'RMSE_F':>7} {'RMSE_D':>7}")
    t1=True
    for c in [0.0,0.4,0.8]:
        r=np.random.default_rng(10+int(c*10)); pf,rf,_,sf=obs_pstar('opposite',c,'F',TGT,PG,MC,r)
        r=np.random.default_rng(10+int(c*10)); pd_,rd,_,sd_=obs_pstar('opposite',c,'D',TGT,PG,MC,r)
        d=abs(pf-pd_) if pf==pf and pd_==pd_ else float('nan')
        print(f"{c:>4} {fmt(pf,sf,PG):>8} {fmt(pd_,sd_,PG):>8} {d:>7.4f} {rf:>7.4f} {rd:>7.4f}")
        if d==d and d>0.015: t1=False
    print(f"T1: {'PASS' if t1 else 'FAIL'}")

    # T3+: 전 위상 × c-sweep
    print("\n### T3+: 전 위상 p*_C(mode,c) — 반전·축별·hold 동시 관측 ###")
    print(f"{'mode':>9} {'c':>4} " + " ".join(f"{t:>8}" for t in TOPOS))
    for mode in ['opposite','scatter']:
        for c in CS:
            cells=[]
            for topo in TOPOS:
                pcs=[]
                for rep in range(REPS):
                    seed=100+rep*997+int(c*10)+(0 if mode=='opposite' else 500)+hash(topo)%50
                    r=np.random.default_rng(seed); pc,rm,tc,st=obs_pstar(mode,c,topo,TGT,PG,MC,r)
                    pcs.append(pc if pc==pc else np.nan)
                    rows.append([mode,c,topo,rep,pc,st,rm])
                arr=np.array(pcs,float)
                val=np.nanmean(arr) if np.isfinite(arr).any() else np.nan
                st_disp='above' if not np.isfinite(arr).any() else 'in'
                cells.append(fmt(val,st_disp,PG))
            print(f"{mode:>9} {c:>4} " + " ".join(f"{x:>8}" for x in cells))
            print(f"    ({time.time()-t0:.0f}s)", end="\r")

    with open('topology_v5_results.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['mode','c','topology','rep','pstar','status','rmse'])
        for row in rows: w.writerow(row)
    print(f"\n\nCSV: topology_v5_results.csv | 총 {time.time()-t0:.0f}s")
    print("판정 지침:")
    print(" · C: opposite에서 전 c 'above'(>0.50)면 정리 A 재현 / scatter에서 c↑ 하강이면 반전 확정")
    print(" · C_vis(대조): zip식 가시성가중 — C와 반대로 opposite가 오히려 낮아야(노출↑)")
    print(" · M_axis vs M_perf: M_axis가 opposite에서 M_perf보다 p* 낮으면 '축별관측=새탐지축'(나-2 지지)")
    print(" · S: 'above'/정상 RMSE면 가드 성공(zip의 168 폭발 수정 확인)")
    print("="*70)
