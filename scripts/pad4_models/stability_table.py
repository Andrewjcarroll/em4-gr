"""R7: 2:1-interface stability table for the methods paper (section 4.1.2).
Block chain: 3 coarse + 6 fine blocks of 24 points (one 2:1 interface each way, periodic) or 9 same-level
blocks. Reports spectral radius of the RK4 amplification matrix, its 500-step 2-norm (transient growth),
and max Re(lambda) of the semi-discrete operator, per scheme / pad / sigma / CFL."""
import numpy as np, csv, chain21 as C, pulse21 as P
rows=[]
cfgs=[('E6',3,None),('JTT6',3,None),('JTT6',4,None),('JTT6',4,3),('A6',3,None),('A6',4,None),('A6',4,3)]
for kind,pad,cfp in cfgs:
    for twoone in (False,True):
        for s in (0.0,0.05,0.1,0.4):
            if twoone: D,x,L=P.build2(kind,pad,3,6,24,cfp); F,_,_=P.build2('KIMF',pad,3,6,24,cfp) if s>0 else (0*D,0,0)
            else:      D,x,L=C.build(kind,pad,0,9,24);     F,_,_=C.build('KIMF',pad,0,9,24) if s>0 else (0*D,0,0)
            A=-D+s*F; mx=np.max(np.linalg.eigvals(A).real)
            for cfl in (0.25,0.5,1.0):
                G=C.rk4_amp(A,cfl); rho=np.max(np.abs(np.linalg.eigvals(G))); g=np.linalg.norm(np.linalg.matrix_power(G,500),2)
                rows.append(dict(scheme=kind,pad=pad,coarse_face_pad=cfp or pad,interface='2:1' if twoone else 'same-level',sigma=s,cfl=cfl,rho=rho,G500=g,maxRe=mx))
                print(f"{kind} pad{pad} cfp{cfp or pad} {'2:1' if twoone else 'same'} s={s:.2f} cfl={cfl}: rho {rho:.6f} G500 {g:9.2f} maxRe {mx:+.2e}",flush=True)
with open('stability_table.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# LaTeX: CFL 0.25 only, rho and G500
with open('stability_table.tex','w') as f:
    f.write('\\begin{tabular}{llcc'+'cc'*4+'}\\toprule\n scheme & pad & interface & & '+' & '.join(f'\\multicolumn{{2}}{{c}}{{$\\sigma={s}$}}' for s in (0,0.05,0.1,0.4))+'\\\\\n')
    f.write(' & & & & '+' & '.join('$\\rho$ & $\\|G^{500}\\|$' for _ in range(4))+'\\\\ \\midrule\n')
    for kind,pad,cfp in cfgs:
        for iface in ('same-level','2:1'):
            sel=[r for r in rows if r['scheme']==kind and r['pad']==pad and r['coarse_face_pad']==(cfp or pad) and r['interface']==iface and r['cfl']==0.25]
            lab=f"{kind} & {pad}{' (3 on coarse faces)' if cfp else ''} & {iface} & "
            f.write(lab+' & '.join(f"{r['rho']:.5f} & {r['G500']:.1f}" for r in sel)+'\\\\\n')
    f.write('\\bottomrule\\end{tabular}\n')
print('written stability_table.csv / .tex')
