"""Rebuild channelAlpha with the channel width of the paper, w = 3*sqrt(U) for an
upstream drainage area U in km2, which is what every network of the paper carries.
Every other array is copied unchanged, and the source file is never overwritten.

    python build/patch_alpha_area.py routing_30min.npz [more.npz ...]

Each file is written back beside itself with _area before the suffix."""
import sys, time, os, numpy as np
BETA, MANNING, SLOPE = 0.6, 0.03, 1e-3
for src_f in sys.argv[1:]:
    t0=time.time()
    res=os.path.basename(src_f)
    dst_f=src_f[:-4]+'_area.npz' if src_f.endswith('.npz') else src_f+'_area.npz'
    z=np.load(src_f); d={k:z[k] for k in z.files}
    s,t,lev=d['edge_src'],d['edge_dst'],d['level']
    acc=(d['cellArea']/1e6).astype(np.float64)          # km2, starts with the cell's own area
    o=np.argsort(lev[s],kind='stable'); s,t=s[o],t[o]
    ls=lev[s]; bounds=np.searchsorted(ls,np.arange(ls.max()+2))
    for L in range(len(bounds)-1):
        a,b=bounds[L],bounds[L+1]
        if b>a: np.add.at(acc,t[a:b],acc[s[a:b]])
    w=3.0*np.sqrt(acc); P=w+2.0
    d['channelAlpha']=((MANNING*P**(2.0/3.0)/np.sqrt(SLOPE))**BETA).astype(d['channelAlpha'].dtype)
    np.savez(dst_f,**d)
    chk=np.median(w/np.sqrt(acc))
    print('%-6s %10d cells  upstream area median %.1f km2  w/sqrt(A)=%.4f  alpha median %.4f -> %.4f  (%.0fs)'
          % (res, lev.size, np.median(acc), chk, float(np.median(z['channelAlpha'])),
             float(np.median(d['channelAlpha'])), time.time()-t0), flush=True)
