"""Put a runoff field on a 0.5-degree grid into the cell order of a routing network,
which is what the routing scripts read.

    python examples/regrid_runoff.py routing_05min.npz qtot_2010.npz runoff_05min.npy

The network npz carries the grid it was built on, the runoff npz holds qtot as
[day, lat, lon] on the global 0.5-degree grid, north to south and from 180 W, and the output is [cell, day] in kg m-2 s-1,
the unit that q = R * cellArea / (1000 * chanLength) expects."""
import sys, numpy as np, time
t0=time.time()
NET, Q, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
d=np.load(NET); mask=d['mask2d']; rows,cols=np.where(~mask)
RES=360.0/mask.shape[1]   # the maps span 360 deg of longitude; CaMa 3' and 1' stop at 60 S
clat=90.0-(rows+0.5)*RES; clon=-180.0+(cols+0.5)*RES
q=np.load(Q); qt=q['qtot']
print('%d routing cells, qtot %s' % (rows.size, qt.shape))
i=np.clip(((90.0-clat)/0.5).astype(int),0,qt.shape[1]-1)
j=np.clip(((clon+180.0)/0.5).astype(int),0,qt.shape[2]-1)
out=np.empty((rows.size, qt.shape[0]), np.float32)
for d_ in range(qt.shape[0]):
    out[:,d_]=qt[d_][i,j]
out=np.nan_to_num(out, nan=0.0)
# units: kg m-2 s-1 (= mm s-1), as q = R*A/(1000*dx) assumes
np.save(OUT, out)
print('wrote %s  %s  %.1f GB  in %.0f s' % (OUT, out.shape, out.nbytes/1e9, time.time()-t0))
