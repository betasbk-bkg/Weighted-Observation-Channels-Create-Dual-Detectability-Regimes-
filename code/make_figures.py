"""
make_figures.py — reproduce all four manuscript figures from the result table.
Usage:  python make_figures.py
Requires: numpy, matplotlib. Reads ../data/topology_v5_results.csv and the
simulator (topology_boundary_pilot_v5.py) in this directory; writes PNGs to ../figures/.
No additional data collection; figures are fully reproducible from the archived table.
"""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import importlib.util

HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,'..','data','topology_v5_results.csv')
FIGS=os.path.join(HERE,'..','figures'); os.makedirs(FIGS,exist_ok=True)
rcParams['font.family']='DejaVu Sans'; rcParams['axes.linewidth']=0.8; rcParams['font.size']=10

from collections import defaultdict
PS=defaultdict(dict); RM=defaultdict(dict); ST=defaultdict(dict)
for r in csv.DictReader(open(DATA)):
    k=(r['mode'],r['topology']); c=float(r['c'])
    ps=float(r['pstar']) if r['pstar'] not in ('','nan') and r['pstar']==r['pstar'] else np.nan
    PS[k][c]=ps; RM[k][c]=float(r['rmse']); ST[k][c]=r['status']
cs=[0.0,0.2,0.4,0.6,0.8,1.0]

# Fig 1: p*(c) reversal
fig,axes=plt.subplots(1,2,figsize=(8,3.4),dpi=150); CAP=0.52
for ax,mode,title in [(axes[0],'opposite','(a) Coherent (opposite) attack'),(axes[1],'scatter','(b) Incoherent (scatter) attack')]:
    for topo,mk,col in [('F','o','#333333'),('D','s','#888888'),('C','^','#c0392b')]:
        y=[];cens=[]
        for c in cs:
            ps=PS[(mode,topo)][c]; st=ST[(mode,topo)][c]
            blind=(ps!=ps or st=='above'); y.append(CAP if blind else ps); cens.append(blind)
        ax.plot(cs,y,marker=mk,color=col,label=topo,lw=1.6,ms=6,mfc='white' if topo=='C' else col)
        for xi,yi,ce in zip(cs,y,cens):
            if ce: ax.annotate('',xy=(xi,CAP+0.015),xytext=(xi,CAP-0.02),arrowprops=dict(arrowstyle='->',color=col,lw=1.2))
    ax.axhline(0.50,ls=':',color='gray',lw=0.8); ax.text(0.02,0.505,'near-majority (0.50)',fontsize=7,color='gray')
    ax.set_xlabel('attack coordination  c'); ax.set_title(title,fontsize=10); ax.set_ylim(0,0.56); ax.set_xlim(-0.05,1.05)
    if ax is axes[0]: ax.set_ylabel('detection boundary  p*')
    ax.legend(frameon=False,fontsize=9,loc='center left'); ax.grid(alpha=0.25,lw=0.5)
axes[1].annotate('↑ censored\n(p*>0.50)',xy=(0.4,0.50),fontsize=7,color='#555',ha='center')
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'fig1_pstar_reversal.png'),bbox_inches='tight'); plt.close()

# Fig 2: variance asymmetry (recompute from simulator)
spec=importlib.util.spec_from_file_location("p5",os.path.join(HERE,'topology_boundary_pilot_v5.py'))
p5=importlib.util.module_from_spec(spec); spec.loader.exec_module(p5)
def gC(v):
    V=p5.DIRS[v]; m=V.mean(0); mh=m/(np.linalg.norm(m)+1e-9)
    w=np.clip(V@mh,0,None)**2; return np.linalg.norm((V*w[:,None]).sum(0)/(w.sum()+1e-9))
cs2=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]); p=0.4; Vo=[];Vs=[]
for c in cs2:
    r=np.random.default_rng(7); Vo.append(np.var([gC(p5.gen_votes(0.,0.,p,'opposite',c,r)) for _ in range(4000)]))
    r=np.random.default_rng(7); Vs.append(np.var([gC(p5.gen_votes(0.,0.,p,'scatter',c,r)) for _ in range(4000)]))
Vo=np.array(Vo);Vs=np.array(Vs)
fig,ax=plt.subplots(figsize=(5,3.6),dpi=150)
ax.plot(cs2,Vs*1e3,'^-',color='#c0392b',label='scatter (incoherent)',lw=1.8,ms=6,mfc='white')
ax.plot(cs2,Vo*1e3,'o-',color='#2c3e50',label='opposite (coherent)',lw=1.8,ms=6)
ax.set_xlabel('attack coordination  c'); ax.set_ylabel(r'Var[$\gamma_C$]  ($\times10^{-3}$)')
ax.set_title(r'Variance-channel sign asymmetry ($p=0.4$)',fontsize=10)
ax.legend(frameon=False,fontsize=9); ax.grid(alpha=0.25,lw=0.5)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'fig2_variance_asymmetry.png'),bbox_inches='tight'); plt.close()

# Fig 3: trade-off
fig,ax=plt.subplots(figsize=(5.2,3.6),dpi=150)
rmse_C=[RM[('opposite','C')][c] for c in cs]; rmse_F=[RM[('opposite','F')][c] for c in cs]
ax.plot(cs,[r/f for r,f in zip(rmse_C,rmse_F)],'D-',color='#27ae60',lw=1.8,ms=6,label='command error ratio C/F')
ax.set_xlabel('attack coordination  c'); ax.set_ylabel('C/F tracking-error ratio',color='#27ae60')
ax.tick_params(axis='y',labelcolor='#27ae60'); ax.set_ylim(0,0.75); ax.axhline(1.0,ls=':',color='gray',lw=0.7)
ax2=ax.twinx()
det=['visible' if ST[('opposite','C')][c]=='in' else 'blind' for c in cs]
ax2.fill_between(cs,0,[1 if d=='blind' else 0 for d in det],color='#c0392b',alpha=0.12,step='mid')
ax2.set_ylim(0,1); ax2.set_yticks([]); ax2.text(0.6,0.5,'detection BLIND\n(p*>0.50)',color='#c0392b',fontsize=8.5,ha='center',alpha=0.8)
ax.set_title('Command\u2013monitor trade-off (opposite attack)',fontsize=10)
ax.legend(frameon=False,fontsize=8.5,loc='upper right'); ax.grid(alpha=0.2,lw=0.5)
plt.tight_layout(); plt.savefig(os.path.join(FIGS,'fig3_tradeoff.png'),bbox_inches='tight'); plt.close()

# Fig 4: heatmap
topos=['F','D','C','C_vis','M_perf','M_axis']
fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),dpi=150)
for ax,mode,ttl in [(axes[0],'opposite','(a) opposite'),(axes[1],'scatter','(b) scatter')]:
    M=np.zeros((len(topos),len(cs)))
    for i,t in enumerate(topos):
        for j,c in enumerate(cs):
            ps=PS[(mode,t)].get(c,np.nan); st=ST[(mode,t)].get(c,'')
            M[i,j]=0.55 if (ps!=ps or st=='above') else ps
    im=ax.imshow(M,aspect='auto',cmap='RdYlGn_r',vmin=0.05,vmax=0.55)
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([f'{c:.1f}' for c in cs],fontsize=8)
    ax.set_yticks(range(len(topos))); ax.set_yticklabels(topos,fontsize=8)
    ax.set_xlabel('c'); ax.set_title(ttl,fontsize=10)
    for i in range(len(topos)):
        for j in range(len(cs)):
            v=M[i,j]; ax.text(j,i,'>.50' if v>=0.55 else f'{v:.2f}',ha='center',va='center',fontsize=6.5)
cbar=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02); cbar.set_label('p*',fontsize=8)
fig.suptitle('Detection boundary by topology and coordination',fontsize=10,y=1.02)
plt.savefig(os.path.join(FIGS,'fig4_heatmap.png'),bbox_inches='tight'); plt.close()
print("All four figures regenerated into", FIGS)
