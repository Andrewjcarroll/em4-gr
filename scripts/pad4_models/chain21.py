#!/usr/bin/env python
"""1D block-chain model of Dendro's unzip with ONE 2:1 refinement interface.

u_t = -u_x on a periodic line: coarse region (spacing 2h) + fine region (spacing h).
Each block computes its derivative from its own points plus `pad` neighbour points:
  same-level face      : neighbour's actual values
  fine block, coarse nb: degree-6 Lagrange interpolant of the adjacent coarse element
  coarse block, fine nb: injection of coincident fine nodes (pad limited to 3 = one
                         fine element; the wide build trims that face to 3)
Derivative: BYU A6 r0.60 Op5 block operator (closures at the padded ends) or E6.
Dissipation: Kim compact filter (KIMF, kc 0.88 pi, eps 0.25) as rhs += (sigma/h)(F-I)u,
built the same way (closures at padded ends). Time: RK4, dt = cfl*h.
Reports spectral radius of the RK4 amplification matrix and ||G^N||_2.
"""
import numpy as np, sys

# ---- BYU_A6_1st_r_060_Op5 -------------------------------------------------
alpha=0.5288235111192885; beta=0.06232711462542066
a1=0.690190101550906; a2=0.19644389104322207; a3=0.00269091403578636
g=dict(g01=10.654929781152678,g02=12.522455029641955,g10=0.025493114185064256,
       g12=3.5316917784950306,g13=2.019628280262894,g20=0.0151011103390593,
       g21=0.30919880265818833,g23=0.7844200188413499,g24=0.1354315949671251)
Pb=[[1.0,g['g01'],g['g02']],[g['g10'],1.0,g['g12'],g['g13']],[g['g20'],g['g21'],1.0,g['g23'],g['g24']]]
Qb=[[-3.808406462522949,-12.682808564359744,11.83255901885571,5.605057071042739,-1.1321193639079083,0.2059282253403907,-0.020209924456862743],
    [-0.14506220847459764,-2.24010711758169,-1.2660731040408477,3.212209799227339,0.4866094760887315,-0.0514602678846433,0.0038834226652815227],
    [-0.06601399450496159,-0.6065930112130826,-0.4441938708782771,0.738100604540617,0.36835328297217473,0.010664600236306922,-0.00031761115278566637]]
Pi=[beta,alpha,1.0,alpha,beta]; Qi=[-a3,-a2,-a1,0.0,a1,a2,a3]

# ---- Kim filter (filt_inmat_kim.h), sigma=1, kc_factor 0.88, eps 0.25 ------
def kim_coeff(kc):
    AF=30-5*np.cos(kc)+10*np.cos(2*kc)-3*np.cos(3*kc)
    return AF, -(30*np.cos(kc)+2*np.cos(3*kc))/AF, (18+9*np.cos(kc)+6*np.cos(2*kc)-np.cos(3*kc))/(2*AF)
def kim_filter(kc_factor=0.88, eps=0.25):
    kc=kc_factor*np.pi
    t2,t3,t6=np.sin(np.pi/2),np.sin(np.pi/3),np.sin(np.pi/6)
    c0=kim_coeff(kc); cd=kim_coeff(kc*(1-eps*t6**2)); cdd=kim_coeff(kc*(1-eps*t3**2)); cddd=kim_coeff(kc*(1-eps*t2**2))
    t1=np.cos(0.5*kc); aF1=30*t1**4/c0[0]; aF2=-2*aF1/5; aF3=aF1/15; aF0=-2*(aF1+aF2+aF3)
    alphaF,betaF=c0[1],c0[2]; alphaFd,betaFd=cd[1],cd[2]; alphaFdd,betaFdd=cdd[1],cdd[2]; alphaFddd,betaFddd=cddd[1],cddd[2]
    t1d=np.cos(0.5*kc*(1-eps*t6**2)); aF1d=30*t1d**4/cd[0]; aF2d=-2*aF1d/5; aF3d=aF1d/15
    BF=(1-betaFdd)*(1+6*betaFdd+60*betaFdd**2)+(5+35*betaFdd-29*betaFdd**2)*alphaFdd+(9-5*betaFdd)*alphaFdd**2
    CF=1+betaFddd*(5+4*betaFddd+60*betaFddd**2)+5*(1+3*betaFddd+10*betaFddd**2)*alphaFddd+2*(4+11*betaFddd)*alphaFddd**2+5*alphaFddd**3
    yF10=(10*betaFdd**2*(8*betaFdd-1)+(1+4*betaFdd+81*betaFdd**2)*alphaFdd+5*(1+8*betaFdd)*alphaFdd**2+9*alphaFdd**3)/BF
    yF20=betaFd
    yF01=(alphaFddd*(1+alphaFddd)*(1+4*alphaFddd)+2*alphaFddd*(7+3*alphaFddd)*betaFddd+24*(1-alphaFddd)*betaFddd**2-80*betaFddd**3)/CF
    yF21=alphaFd
    yF02=(alphaFddd**3+(1+3*alphaFddd+14*alphaFddd**2)*betaFddd+46*alphaFddd*betaFddd**2+60*betaFddd**3)/CF
    yF12=(alphaFdd*(1+5*alphaFdd+9*alphaFdd**2)+alphaFdd*(5+36*alphaFdd)*betaFdd+(55*alphaFdd-1)*betaFdd**2+10*betaFdd**3)/BF
    yF13=betaFdd*(1+5*alphaFdd+9*alphaFdd**2+5*(1+7*alphaFdd)*betaFdd+50*betaFdd**2)/BF
    yF23=alphaFd; yF24=betaFd
    bF20=aF2d+5*aF3d; bF21=aF1d-10*aF3d; bF23=aF1d-5*aF3d; bF24=aF2d+aF3d; bF25=aF3d; bF22=-(bF20+bF21+bF23+bF24+bF25)
    Ri=[betaF,alphaF,1.0,alphaF,betaF]
    Rb=[[1.0,yF01,yF02],[yF10,1.0,yF12,yF13],[yF20,yF21,1.0,yF23,yF24]]
    Si=[aF3,aF2,aF1,aF0,aF1,aF2,aF3]
    Sb=[[0,0,0,0,0],[0,0,0,0,0],[bF20,bF21,bF22,bF23,bF24,bF25]]
    return Ri,Rb,Si,Sb

def band(n, interior, bnd, parity):
    """n x n matrix: closure rows `bnd` at top (cols from 0), mirrored at bottom with parity, interior band elsewhere."""
    M=np.zeros((n,n)); nb=len(bnd); hw=len(interior)//2
    for i in range(n):
        if i<nb:
            for j,v in enumerate(bnd[i]): M[i,j]=v
        elif i>=n-nb:
            r=n-1-i
            for j,v in enumerate(bnd[r]): M[i,n-1-j]=parity*v
        else:
            for k,v in enumerate(interior): M[i,i-hw+k]=v
    return M

def block_op(n, kind, pl, pr):
    """n-point padded block operator (rows = all n; caller keeps active rows). Trim = identity row at
    pad index 0 when the pad on that side is only 3 in a pad-4 build: we model that by just using pl/pr."""
    if kind=='A6':
        P=band(n,Pi,Pb,+1.0); Q=band(n,Qi,Qb,-1.0); return np.linalg.solve(P,Q)
    if kind=='E6':
        c=np.array([-1,9,-45,0,45,-9,1])/60.0; M=np.zeros((n,n))
        for i in range(3,n-3): M[i,i-3:i+4]=c
        return M
    if kind=='KIMF':
        Ri,Rb,Si,Sb=kim_filter(); R=band(n,Ri,Rb,+1.0); S=band(n,Si,Sb,+1.0); return np.linalg.solve(R,S)
    raise ValueError(kind)

def lagrange_weights(xs, x):
    xs=np.asarray(xs,float); w=np.ones(len(xs))
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i!=j: w[i]*=(x-xs[j])/(xs[i]-xs[j])
    return w

def build(kind, pad, nblk_c=3, nblk_f=6, m=24, h=1.0, deriv=True):
    """Global operator (units 1/h) on periodic line: coarse blocks (spacing 2h) then fine blocks (spacing h).
    Returns matrix acting on u = [coarse..., fine...]. Coarse pad from fine = injection (max 3 points);
    fine pad from coarse = Lagrange-6 on the adjacent coarse element's 7 nodes (uniform, spacing 2h)."""
    Nc=nblk_c*m; Nf=nblk_f*m; N=Nc+Nf
    xc=np.arange(Nc)*2*h; xf=Nc*2*h+np.arange(Nf)*h; x=np.concatenate([xc,xf]); L=Nc*2*h+Nf*h
    Lg=np.zeros((N,N))
    # source maps: pad value = row vector over u
    def same(idx): e=np.zeros(N); e[idx%N]=1; return e
    for b in range(nblk_c+nblk_f):
        coarse=b<nblk_c; hh=2*h if coarse else h
        lo=b*m; pts=list(range(lo,lo+m))
        # left pad
        left=[]; pl=pad
        if coarse and b==0:            # left neighbour is the last FINE block (periodic)
            pl=min(pad,3); left=[same(N-1-(2*k+1)) for k in range(pl)][::-1]  # coincident fine nodes: x = -2h,-4h,...
            # coincident fine node with coarse position -2h(k+1): fine index Nf-1-(2k+1)... fine x = L - h*(j+1) for j from end
            left=[]
            for k in range(pl,0,-1):   # coarse pad positions -2h*k
                xt=-2*h*k+L; j=int(round((xt-xf[0])/h)); left.append(same(Nc+j))
        elif not coarse and nblk_c>0 and b==nblk_c: # left neighbour is coarse: Lagrange on the adjacent coarse element (last 7 coarse nodes)
            pl=pad; xs=np.concatenate([xc[-6:],[xf[0]]]); left=[]   # element ending at the shared face node (= fine index 0)
            for k in range(pl,0,-1):
                xt=xf[0]-h*k; w=lagrange_weights(xs,xt); e=np.zeros(N); e[Nc-6:Nc]=w[:6]; e[Nc]=w[6]; left.append(e)
        else:
            left=[same(lo-k) for k in range(pl,0,-1)]
        right=[]; pr=pad
        if coarse and b==nblk_c-1:     # right neighbour is fine: injection
            pr=min(pad,3); right=[]
            for k in range(1,pr+1):
                xt=xc[-1]+2*h*k; j=int(round((xt-xf[0])/h)); right.append(same(Nc+j))
        elif not coarse and nblk_c>0 and b==nblk_c+nblk_f-1:  # right neighbour is coarse (periodic): Lagrange on first coarse element
            pr=pad; xs=xc[:7]+L; right=[]
            for k in range(1,pr+1):
                xt=xf[-1]+h*k; w=lagrange_weights(xs,xt); e=np.zeros(N); e[0:7]=w; right.append(e)
        else:
            right=[same(lo+m-1+k) for k in range(1,pr+1)]
        n=pl+m+pr
        srcs=left+[same(p) for p in pts]+right
        S=np.array(srcs)                  # n x N
        D=block_op(n,kind,pl,pr)          # n x n, units of 1/spacing
        Dact=D[pl:pl+m,:]/hh
        Lg[lo:lo+m,:]=Dact@S
    return Lg, x, L

def rk4_amp(A, dt):
    N=A.shape[0]; M=dt*A; G=np.eye(N)+M+M@M/2+M@M@M/6+M@M@M@M/24; return G

def run(kind, pad, sigma, cfl=0.25, nsteps=500, twoone=True, m=24):
    if twoone: D,x,L=build(kind,pad,3,6,m)
    else:      D,x,L=build(kind,pad,0,9,m)
    if sigma>0:
        F,_,_=build('KIMF',pad,3,6,m) if twoone else build('KIMF',pad,0,9,m)
    else: F=0*D
    A=-D+sigma*F     # h=1
    dt=cfl
    G=rk4_amp(A,dt)
    ev=np.linalg.eigvals(G); rho=np.max(np.abs(ev))
    Gn=np.linalg.matrix_power(G,nsteps); nrm=np.linalg.norm(Gn,2)
    evA=np.linalg.eigvals(A); mx=np.max(evA.real)
    return rho, nrm, mx

if __name__=='__main__':
    for kind in ['E6','A6']:
        D,x,L=build(kind,3,3,6,24); u=np.sin(2*np.pi*3*x/L); du=2*np.pi*3/L*np.cos(2*np.pi*3*x/L)
        err=np.abs(D@u-du); Nc=72
        print(f"consistency {kind}: max err coarse {err[:Nc].max():.2e} fine {err[Nc:].max():.2e}  near-face coarse {err[Nc-8:Nc].max():.2e} fine {err[Nc:Nc+8].max():.2e}")
    print("kind pad sigma 2:1 | rho(G) ||G^500||  max Re(lambda_A)  (units 1/h, dt=0.25h)")
    for kind in ['E6','A6']:
        for pad in ([3] if kind=='E6' else [3,4]):
            for sigma in [0.0,0.05,0.1,0.4]:
                for twoone in [False,True]:
                    rho,nrm,mx=run(kind,pad,sigma,twoone=twoone)
                    print(f"{kind:3s} {pad}  {sigma:4.2f}  {'yes' if twoone else 'no '} | {rho:.6f}  {nrm:10.3e}  {mx:+.3e}")
