import numpy as np, chain21 as C
# add an option: fine block next to a coarser neighbour uses only 3 (trimmed) prolongated pad points
def build2(kind,pad,nblk_c,nblk_f,m,coarse_face_pad=None):
    if coarse_face_pad is None: return C.build(kind,pad,nblk_c,nblk_f,m)
    # monkeypatch: temporarily shrink pad on the fine-from-coarse faces
    src=C.build.__code__
    import types
    N=nblk_c*m+nblk_f*m; h=1.0
    Nc=nblk_c*m; Nf=nblk_f*m
    xc=np.arange(Nc)*2*h; xf=Nc*2*h+np.arange(Nf)*h; L=Nc*2*h+Nf*h
    Lg=np.zeros((N,N))
    def same(idx): e=np.zeros(N); e[idx%N]=1; return e
    for b in range(nblk_c+nblk_f):
        coarse=b<nblk_c; hh=2*h if coarse else h; lo=b*m; pts=list(range(lo,lo+m))
        pl=pad; pr=pad
        if coarse and b==0:
            pl=min(pad,3); left=[same(Nc+int(round((-2*h*k+L-xf[0])/h))) for k in range(pl,0,-1)]
        elif not coarse and b==nblk_c:
            pl=coarse_face_pad; xs=np.concatenate([xc[-6:],[xf[0]]]); left=[]
            for k in range(pl,0,-1):
                w=C.lagrange_weights(xs,xf[0]-h*k); e=np.zeros(N); e[Nc-6:Nc]=w[:6]; e[Nc]=w[6]; left.append(e)
        else: left=[same(lo-k) for k in range(pl,0,-1)]
        if coarse and b==nblk_c-1:
            pr=min(pad,3); right=[same(Nc+int(round((xc[-1]+2*h*k-xf[0])/h))) for k in range(1,pr+1)]
        elif not coarse and b==nblk_c+nblk_f-1:
            pr=coarse_face_pad; xs=xc[:7]+L; right=[]
            for k in range(1,pr+1):
                w=C.lagrange_weights(xs,xf[-1]+h*k); e=np.zeros(N); e[0:7]=w; right.append(e)
        else: right=[same(lo+m-1+k) for k in range(1,pr+1)]
        n=pl+m+pr; S=np.array(left+[same(p) for p in pts]+right)
        D=C.block_op(n,kind,pl,pr); Lg[lo:lo+m,:]=D[pl:pl+m,:]/hh@S
    return Lg, np.concatenate([xc,xf]), L

def experiment(kind,pad,sigma,cfp=None,nblk_c=4,nblk_f=12,m=24,cfl=0.25,width=21.0,start='coarse'):
    D,x,L=build2(kind,pad,nblk_c,nblk_f,m,cfp)
    F,_,_=build2('KIMF',pad,nblk_c,nblk_f,m,cfp) if sigma>0 else (0*D,None,None)
    A=-D+sigma*F; dt=cfl; Nc=nblk_c*m
    G=C.rk4_amp(A,dt)
    x0 = (x[Nc]-60.0) if start=='coarse' else (x[Nc]+60.0)   # pulse centre 60h before/after the interface
    def exact(t):
        d=(x-x0-t+L/2)%L-L/2; return np.exp(-(d/(width/2.355))**2/2)   # width = FWHM in h units
    u=exact(0.0); errs=[]
    nsteps=int(round(125/dt)) if start=='coarse' else int(round(125/dt))
    for s in range(1,nsteps+1):
        u=G@u
        if s in (int(round(40/dt)),int(round(80/dt)),nsteps):
            e=u-exact(s*dt); errs.append((s*dt, np.sqrt(np.mean(e[Nc:]**2)), np.sqrt(np.mean(e[:Nc]**2)), np.abs(e).max()))
    return errs
print("pulse FWHM 21h; centre starts 60h before the coarse->fine face, advects 125h (crosses into fine at t~60)")
print("kind pad cfp sigma | t=40 rmsF rmsC | t=80 rmsF rmsC | t=125 rmsF rmsC | max|e| end")
for kind,pad,cfp in [('E6',3,None),('A6',3,None),('A6',4,None),('A6',4,3)]:
    for sigma in [0.0,0.05,0.1,0.4]:
        er=experiment(kind,pad,sigma,cfp)
        print(f"{kind} {pad} {str(cfp):4s} {sigma:4.2f} | "+" | ".join(f"{a[1]:.2e} {a[2]:.2e}" for a in er)+f" | {er[-1][3]:.2e}")
print("--- pulse starts 60h AFTER the face (inside fine), advects 125h: crosses fine->coarse at the periodic wrap (~t=165)?? no: fine region is 288h long, so it stays fine; this isolates same-level+filter effects")
for kind,pad,cfp in [('E6',3,None),('A6',3,None),('A6',4,None)]:
    for sigma in [0.0,0.1,0.4]:
        er=experiment(kind,pad,sigma,cfp,start='fine')
        print(f"{kind} {pad} {str(cfp):4s} {sigma:4.2f} | "+" | ".join(f"{a[1]:.2e} {a[2]:.2e}" for a in er)+f" | {er[-1][3]:.2e}")
