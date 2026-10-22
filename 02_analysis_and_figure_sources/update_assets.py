"""Rebuild only figures/tables affected by the rank-local RNG implementation."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
from matplotlib.ticker import LogLocator
import retained_plot_helpers as p

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=HERE/"data"
MAN=ROOT/"01_manuscript_tex_and_source"
p.DATA=DATA/"retained"
p.MAN=MAN
p.HERE=HERE
p.plt.rcParams.update({"font.size":9,"axes.labelsize":9.5,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8.5})
TIMING=p.read_csv(DATA/"timing_records.csv")
SEEDS=p.SEEDS
STATS=[]
PAIRS=[]


def values(path):return [{k:float(v) for k,v in r.items() if v!=""} for r in p.read_csv(path)]


def trajectory(case,kind,rank=24,seed=SEEDS[0],exact=False,dynamic=False,half=False):
    if exact:
        path=DATA/"retained/core/snapshots"/f"exact__{case}__p1__s{seed}__{kind}.csv"
        data=values(path);xkey="attempted_sweeps"
    elif dynamic and rank==1:
        path=DATA/"retained/incremental/snapshots"/f"dynamic__{case}__p1__s{seed}__{kind}__dt{'half' if half else 'base'}.csv"
        data=values(path);every=200 if half else 100
        data=[r for r in data if int(r["local_md_step"])%every==0]
        xkey="sweeps"
    else:
        prefix="mdmc" if dynamic else "parallel"
        path=DATA/"recomputed/timeseries"/f"{prefix}__{case}__p{rank}__s{seed}__{kind}.csv"
        data=values(path);xkey="sweeps"
    return np.array([r[xkey] for r in data]),data


def repeated(case,rank,n,kind,cohort="legacy"):
    rows=[r for r in TIMING if r["case"]==case and int(r["ranks"])==rank and int(r["natoms"])==n
          and r["kind"]==kind and r["cohort"]==cohort]
    rows=sorted(rows,key=lambda r:int(r["repeat"]))
    expected=5 if cohort=="size3d" else 3
    if len(rows)!=expected:raise ValueError(f"Mixed/missing timing cohort: {case}/{rank}/{n}/{kind}/{cohort}: {len(rows)}")
    return np.array([float(r["time_per_actual_sweep_s"]) for r in rows])


def ratio(a,b):
    point=float(a.mean()/b.mean())
    if np.array_equal(a,b):return point,point,point
    # Historical reference and newly timed fast records are not contemporaneous blocks.
    se=math.sqrt(a.var(ddof=1)/len(a)/a.mean()**2+b.var(ddof=1)/len(b)/b.mean()**2)
    half=float(t.ppf(.975,min(len(a),len(b))-1))*se
    return point,point*math.exp(-half),point*math.exp(half)


def correlated(x):
    mean=float(np.mean(x));constant=bool(np.ptp(x)<=128*np.finfo(float).eps*max(1,abs(mean)))
    if constant:return dict(mean=mean,se=0.,ess=0.,df=1,constant=True)
    y=x-mean;length=1<<(2*len(x)-1).bit_length();ft=np.fft.rfft(y,length)
    acf=np.fft.irfft(ft*ft.conjugate(),length)[:len(x)];acf/=acf[0]
    positive=0.
    for i in range(1,len(x)-1,2):
        z=acf[i]+acf[i+1]
        if z<=0:break
        positive+=z
    ess=min(len(x),len(x)/max(1,1+2*positive));se=float(x.std(ddof=1)/math.sqrt(ess));df=max(1,int(ess)-1)
    for width in (1,2,4,8,16,32,64,128):
        nb=len(x)//width
        if nb>=4:
            z=x[-nb*width:].reshape(nb,width).mean(axis=1);block=float(z.std(ddof=1)/math.sqrt(nb))
            if block>se:se,df=block,nb-1
    return dict(mean=mean,se=se,ess=float(ess),df=df,constant=False)


def stats(case,rank,obs,dynamic=False,half=False,discard=.25):
    rows=[]
    for kind in ("atom","fast"):
        for seed in SEEDS:
            x,data=trajectory(case,kind,rank,seed,dynamic=dynamic,half=half)
            selected=np.array([r[obs] for r,xx in zip(data,x) if xx>=discard*x[-1]])
            rows.append(dict(case=case,ranks=rank,kind=kind,seed=seed,half_dt=half,discard=discard,observable=obs,
                             family="mdmc" if dynamic else "parallel",samples=len(selected),**correlated(selected)))
    aa=[r for r in rows if r["kind"]=="atom"];bb=[r for r in rows if r["kind"]=="fast"]
    differences=np.array([b["mean"]-a["mean"] for a,b in zip(aa,bb)])
    se=max(float(differences.std(ddof=1)/math.sqrt(3)),math.sqrt(sum(r["se"]**2 for r in rows))/3)
    degrees=min(2,*(r["df"] for r in rows))
    pair=dict(case=case,ranks=rank,half_dt=half,discard=discard,observable=obs,family="mdmc" if dynamic else "parallel",
              difference=float(differences.mean()),ci95=float(t.ppf(.975,degrees)*se),minimum_ess=min(r["ess"] for r in rows),
              constant=all(r["constant"] for r in rows))
    STATS.extend(rows);PAIRS.append(pair)
    return rows,pair


def cohort(kind,rows):
    values=[r for r in rows if r["kind"]==kind];means=np.array([r["mean"] for r in values])
    se=max(float(means.std(ddof=1)/math.sqrt(3)),math.sqrt(sum(r["se"]**2 for r in values))/3)
    return float(means.mean()),float(t.ppf(.975,min(2,*(r["df"] for r in values)))*se)


def csvsave(name,rows):
    path=DATA/name
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def tables():
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Model & Quantity & atom/swap & fast/swap & Difference & Min. ESS \\",r"\midrule"]
    for case in p.ORDER:
        for obs in (["U_per_atom","x_type_2","x_type_3"] if case=="ternary_lj_cut" else ["U_per_atom","x_type_2"]):
            rows,pair=stats(case,24,obs);a,ac=cohort("atom",rows);b,bc=cohort("fast",rows)
            if pair["constant"]:
                cells=[f"${a:.6f}$",f"${b:.6f}$","$"+p.sci(pair["difference"])+"$","--"]
            else:
                cells=[f"${a:.6f}\\pm {p.sci(ac,2)}$",f"${b:.6f}\\pm {p.sci(bc,2)}$",
                       "$"+p.sci(pair["difference"],3)+r"\pm "+p.sci(pair["ci95"],2)+"$",f"{pair['minimum_ess']:.0f}"]
            label=p.NAME[case] if obs=="U_per_atom" else ""
            quantity=r"$U/N$" if obs=="U_per_atom" else (r"$x_B$" if obs=="x_type_2" else r"$x_C$")
            lines.append(label+" & "+quantity+" & "+" & ".join(cells)+r" \\")
            stats(case,24,obs,discard=.5)
    lines += [r"\bottomrule",r"\end{tabular}"]
    (MAN/"revision_parallel_table.tex").write_text("\n".join(lines)+"\n")
    oldaccept=p.read_csv(DATA/"retained/audit/acceptance_rechecked.csv")
    acce=[];lines=[r"\begin{tabular}{lrrr}",r"\toprule",r"Model & Sweeps & atom/swap & fast/swap \\",r"\midrule"]
    for case in p.ORDER:
        cells=[]
        for kind in ("atom","fast"):
            if kind=="atom":
                v=[float(r["acceptance"]) for r in oldaccept if r["case"]==case and r["ranks"]=="24" and r["kind"]==kind and r["discard"]=="0.25"]
            else:
                v=[]
                for seed in SEEDS:
                    x,data=trajectory(case,kind,seed=seed);window=[r for r,xx in zip(data,x) if xx>=.25*x[-1]]
                    v.append((window[-1]["accepts"]-window[0]["accepts"])/(window[-1]["attempts"]-window[0]["attempts"]))
            assert len(v)==3
            acce += [dict(case=case,kind=kind,seed=s,acceptance=a) for s,a in zip(SEEDS,v)]
            cells.append(f"{np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]")
        lines.append(p.NAME[case]+f" & {1200 if case in ('zbl','ternary_lj_cut') else 600} & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (MAN/"revision_acceptance_table.tex").write_text("\n".join(lines)+"\n")
    csvsave("acceptance_updated.csv",acce)
    lines=[r"\begin{tabular}{llrrr}",r"\toprule",r"Model & Protocol & Quantity & Difference $\pm$ 95\% CI & Min. ESS \\",r"\midrule"]
    for case in ("lj_cut","morse_dynamic"):
        for rank,half in ((1,False),(2,False),(1,True)):
            for obs in ("U_per_atom","x_type_2"):
                _,pair=stats(case,rank,obs,dynamic=True,half=half)
                label=p.NAME[case] if rank==1 and not half and obs=="U_per_atom" else ""
                protocol=f"{rank} rank"+("s" if rank>1 else "")+(r", $\Delta t/2$" if half else "")
                q=r"$U/N$" if obs=="U_per_atom" else r"$x_B$"
                lines.append(label+" & "+protocol+" & "+q+" & $"+p.sci(pair["difference"],3)+r"\pm "+p.sci(pair["ci95"],2)+f"$ & {pair['minimum_ess']:.1f}"+r" \\")
                stats(case,rank,obs,dynamic=True,half=half,discard=.5)
    lines += [r"\bottomrule",r"\end{tabular}"]
    (MAN/"revision_dynamic_table.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{llrr}",r"\toprule",
           r"Model & Implementation & Quantity & $\Delta t/2-\Delta t$ mean difference $\pm$ 95\% CI \\",r"\midrule"]
    controls=[]
    for case in ("lj_cut","morse_dynamic"):
        for kind in ("atom","fast"):
            for obs in ("U_per_atom","x_type_2"):
                differences=[];within=[];dfs=[]
                for seed in SEEDS:
                    chain=[]
                    for half in (False,True):
                        x,data=trajectory(case,kind,1,seed,dynamic=True,half=half)
                        values=np.array([r[obs] for xx,r in zip(x,data) if 75<=xx<=300])
                        chain.append(correlated(values))
                    differences.append(chain[1]["mean"]-chain[0]["mean"])
                    within.extend(z["se"]**2 for z in chain);dfs.extend(z["df"] for z in chain)
                mean=float(np.mean(differences))
                se=max(float(np.std(differences,ddof=1)/math.sqrt(3)),math.sqrt(sum(within))/3)
                ci=float(t.ppf(.975,min(2,*dfs))*se)
                controls.append(dict(case=case,kind=kind,observable=obs,difference=mean,ci95=ci))
                quantity=r"$U/N$" if obs=="U_per_atom" else r"$x_B$"
                lines.append(p.NAME[case]+" & "+kind+"/swap & "+quantity+" & $"+p.sci(mean,3)+r"\pm "+p.sci(ci,2)+r"$ \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (MAN/"revision_timestep_table.tex").write_text("\n".join(lines)+"\n")
    csvsave("timestep_comparisons_updated.csv",controls)
    csvsave("replica_statistics_updated.csv",STATS);csvsave("mean_comparisons_updated.csv",PAIRS)


def size_plot():
    fig,(a,b)=p.plt.subplots(2,1,figsize=(3.55,4.75))
    guides=[]
    for rank,scope,marker,ls in ((1,"legacy","o","-"),(24,"size3d","s","--")):
        ns=sorted({int(r["natoms"]) for r in TIMING if r["case"]=="lj_cut" and r["cohort"]==scope and int(r["ranks"])==rank and int(r["natoms"])!=24576})
        for kind,color in (("atom",p.BLUE),("fast",p.ORANGE)):
            y,err=np.array([p.uncertainty(repeated("lj_cut",rank,n,kind,scope)) for n in ns]).T
            a.errorbar(ns,y,yerr=err,color=color,marker=marker,ls=ls,ms=3.3,capsize=2,lw=.9,zorder=3)
            power=2 if kind=="atom" else 1
            offset=1.65 if kind=="atom" else .62
            coefficient=offset*np.exp(np.mean(np.log(y)-power*np.log(ns)))
            gx=np.geomspace(ns[0],ns[-1],100)
            a.plot(gx,coefficient*gx**power,color="#303030",
                   ls=(0,(5,2.5)) if power==2 else (0,(1.5,1.6)),lw=1.25,zorder=2)
            label_x=(1150 if rank==1 else 14500) if power==2 else (3500 if rank==1 else 17000)
            a.annotate(r"$N^2$" if power==2 else r"$N$",
                       xy=(label_x,coefficient*label_x**power),
                       xytext=(1,5 if power==2 else -12),textcoords="offset points",
                       fontsize=8.5,color="#202020",ha="center",va="bottom",
                       bbox=dict(facecolor="white",edgecolor="none",pad=.3,alpha=.94))
            guides.append(dict(ranks=rank,energy_path=kind,power=power,
                               coefficient=float(coefficient),normalization_offset=offset,
                               minimum_N=ns[0],maximum_N=ns[-1],is_fit=False))
        y,lo,hi=np.array([ratio(repeated("lj_cut",rank,n,"atom",scope),repeated("lj_cut",rank,n,"fast",scope)) for n in ns]).T
        b.errorbar(ns,y,yerr=[y-lo,hi-y],color=p.BLUE if rank==1 else p.ORANGE,marker=marker,ls=ls,ms=3.5,capsize=2,lw=.9)
    a.set_yscale("log")
    a.yaxis.set_major_locator(LogLocator(base=10,numticks=9))
    for ax,letter in ((a,"a"),(b,"b")):
        p.panel(ax,letter);ax.set_xscale("log");ax.set_xlim(420,39000)
        ax.set_xticks([500,1000,3000,10000,30000]);ax.xaxis.set_major_formatter(p.FuncFormatter(lambda x,_:f"{int(x):,}"))
        ax.set_xlabel("Number of atoms, $N$")
    a.set_ylabel("Wall time per sweep (s)");b.set_ylabel("Same-rank speedup")
    b.set_ylim(0,1200);b.set_yticks(np.arange(0,1201,200))
    b.legend(handles=[p.Line2D([],[],color=p.BLUE,marker="o",label="1 rank"),
                      p.Line2D([],[],color=p.ORANGE,marker="s",ls="--",label="24 ranks")],
             loc="lower right",bbox_to_anchor=(1,1.04),ncol=2,frameon=False,fontsize=7.7)
    hs=[p.Line2D([],[],color=p.BLUE,marker="o",label="atom, 1 rank"),p.Line2D([],[],color=p.ORANGE,marker="o",label="fast, 1 rank"),
        p.Line2D([],[],color=p.BLUE,marker="s",ls="--",label="atom, 24 ranks"),p.Line2D([],[],color=p.ORANGE,marker="s",ls="--",label="fast, 24 ranks")]
    fig.subplots_adjust(left=.19,right=.97,bottom=.10,top=.84,hspace=.52)
    fig.legend(handles=hs,loc="upper center",bbox_to_anchor=(.54,1),ncol=2,frameon=False,fontsize=7.9)
    p.save(fig,"Figure_4","Unchanged serial/3D data; darker offset N and N^2 guides labelled directly, not fitted slopes.")
    (HERE/"figure4_guide_audit.json").write_text(json.dumps(guides,indent=2)+"\n")


def main():
    p.trajectory=trajectory;p.repeated=repeated;p.ratio=ratio
    tables();p.main_validation();p.main_mobile();size_plot();p.main_strong();p.main_styles()
    p.dense_trajectories("Figure_S3",["morse","table_lj","buck","buck_coul_cut","coul_dsf","hybrid_overlay_lj_yukawa","yukawa","zbl"])
    p.si_lj_variants();p.si_nonlj();p.si_unequal();p.si_repeats()
    p.dense_trajectories("Figure_S8",[("lj_cut",1,False),("lj_cut",2,False),("morse_dynamic",1,False),("morse_dynamic",2,False),("lj_cut",1,True),("morse_dynamic",1,True)],dynamic=True)
    (HERE/"figure_index.json").write_text(json.dumps(p.ASSETS,indent=2)+"\n")
    (HERE/"figure_layout_checks.json").write_text(json.dumps(p.CHECKS,indent=2)+"\n")
    print("Updated",len(p.ASSETS),"figures and statistical tables; retained exact/closure and application assets")


if __name__=="__main__":
    import sys
    if "--figure4-only" in sys.argv:
        size_plot()
        for name,entries in (("figure_index.json",p.ASSETS),("figure_layout_checks.json",p.CHECKS)):
            path=HERE/name
            old=json.loads(path.read_text()) if path.exists() else []
            keys={row["file"] for row in entries}
            merged=[row for row in old if row["file"] not in keys]+entries
            path.write_text(json.dumps(merged,indent=2)+"\n")
    else:main()
