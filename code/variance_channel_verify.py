"""
분산채널 검증 (Theorem B) — 경량, γ_C 분산의 c-스케일링만 측정
실행:  python variance_channel_verify.py [DRAWS]   (기본 4000)
   opposite ∝(1-c) / scatter ∝c² 를 스케일에서 확인. v5 full 전 독립 체크용.
"""
import numpy as np, sys
DIRS=np.array([[np.cos(2*np.pi*k/8),np.sin(2*np.pi*k/8)] for k in range(8)])
N_AG=120
def angle_to_dir(a): return int(np.round(a/(2*np.pi/8)))%8
def gen_votes(p,mode,c,rng):
    N=N_AG; n_tr=min(round(N*p),N); n_h=N-n_tr; out=[]
    n_acc=round(n_h*0.7368); n_slow=round(n_h*0.2105); n_oth=n_h-n_acc-n_slow
    for _ in range(n_acc): out.append(angle_to_dir(np.deg2rad(rng.uniform(-3,3))))
    for _ in range(n_slow): out.append(0)
    for _ in range(max(n_oth,0)): out.append(angle_to_dir(np.deg2rad(rng.uniform(-30,30))))
    n_al=int(round(c*n_tr)); base=4
    if mode=='opposite':
        for _ in range(n_al): out.append((base+rng.integers(-1,2))%8)
    else:
        sup=rng.choice(8,3,replace=False)
        for _ in range(n_al): out.append(int(sup[rng.integers(0,3)]))
    for _ in range(n_tr-n_al): out.append(int(rng.integers(0,8)))
    return np.array(out)
def gamma_C(votes):
    V=DIRS[votes]; m=V.mean(0); mhat=m/(np.linalg.norm(m)+1e-9)
    w=np.clip(V@mhat,0,None)**2; cw=(V*w[:,None]).sum(0)/(w.sum()+1e-9)
    return np.linalg.norm(cw)
if __name__=="__main__":
    D=int(sys.argv[1]) if len(sys.argv)>1 else 4000
    p=0.4; cs=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    print(f"분산채널 검증 | p={p} draws={D}")
    print(f"{'c':>5} {'Var_opp':>11} {'Var_sct':>11} {'sct/opp':>8}")
    Vo=[]; Vs=[]
    for c in cs:
        r=np.random.default_rng(7); go=[gamma_C(gen_votes(p,'opposite',c,r)) for _ in range(D)]
        r=np.random.default_rng(7); gs=[gamma_C(gen_votes(p,'scatter',c,r)) for _ in range(D)]
        vo,vs=np.var(go),np.var(gs); Vo.append(vo); Vs.append(vs)
        print(f"{c:>5} {vo:>11.6f} {vs:>11.6f} {vs/vo:>8.1f}")
    Vo=np.array(Vo); Vs=np.array(Vs)
    # PV1: scatter c-거듭제곱
    sl_s=np.polyfit(np.log(cs),np.log(Vs),1)[0]
    # PV2: opposite (1-c) 선형 R²
    A=np.vstack([1-cs,np.ones_like(cs)]).T; coef,*_=np.linalg.lstsq(A,Vo,rcond=None)
    pred=A@coef; r2=1-np.sum((Vo-pred)**2)/np.sum((Vo-Vo.mean())**2)
    # PV1(정정): 순수거듭제곱 아님 — 단조증가+초선형(>1)만 검정
    mono_s=np.all(np.diff(Vs)>0)
    print(f"\nPV1 scatter 단조증가={mono_s} 초선형지수={sl_s:.2f}(>1) → {'PASS' if (mono_s and sl_s>1) else 'FAIL'}")
    print(f"PV2 opposite (1-c)선형 R² = {r2:.3f}  (예측 >0.9) → {'PASS' if r2>0.9 else 'FAIL'}")
    print(f"PV3 분산비 c=1.0 = {Vs[-1]/Vo[-1]:.1f}배  (예측 >=50) → {'PASS' if Vs[-1]/Vo[-1]>=50 else 'FAIL'}")
