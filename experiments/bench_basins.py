"""The 26 large river basins of Nijssen et al. (2001), located by river mouth.

Each entry is (river, outlet latitude, outlet longitude). Volga and Brahmaputra are cut
at a point inside a larger CaMa basin rather than at a mouth. The outlet is the
highest-drainage cell within half a degree of the point (subset_basin.py).

Used by bench_unified_timing.py time-basins, which cuts each basin out of the 1 arc-minute
network and times SURGE and CPU-serial on it (Figure 8), and by basin_k_from_year.py.
"""

BASINS = [
    ('Amazon', -0.5, -50.7), ('Congo', -6.0, 12.4), ('Danube', 45.2, 29.7),
    ('Mackenzie', 69.0, -136.0), ('Mekong', 9.6, 106.3), ('Mississippi', 29.0, -89.4),
    ('Senegal', 15.9, -16.5), ('Yellow', 37.8, 119.2), ('Yenisei', 71.0, 83.3),
    ('Amur', 52.9, 141.2), ('Brahmaputra', 25.2, 89.7), ('Changjiang', 31.8, 121.1),
    ('Columbia', 46.17, -123.27), ('Indigirka', 71.0, 150.0), ('Irrawaddy', 16.0, 95.0),
    ('Kolyma', 69.5, 161.5), ('Lena', 72.0, 129.1), ('Ob', 66.5, 69.8),
    ('Olenek', 73.0, 120.0), ('Parana', -34.1, -58.3), ('Pechora', 68.2, 54.0),
    ('S.Dvina', 64.6, 40.5), ('Volga', 46.3, 48.0), ('Xi', 22.5, 113.5),
    ('Yana', 71.5, 136.0), ('Yukon', 63.0, -164.6),
]
